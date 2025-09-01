# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json
import random
import time
import math
from collections import defaultdict, Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypedDict, List
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import (
    ClippedPGLossConfig,
    ClippedPGLossDataDict,
    ClippedPGLossFn,
)
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, rl_collate_fn
from nemo_rl.data.interfaces import (
    DatumSpec,
)
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
)
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
)
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.interfaces import PolicyInterface
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.hf_policy import HfPolicy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.timer import Timer


# ===============================================================================
# Helper Functions  
# ===============================================================================

# Configuration example for baseline plan:
# "plan_grpo": {
#     "use_baseline_plan": true,
#     "baseline_plan_template": "Carefully evaluate both responses by:\n1. Checking factual accuracy\n2. Assessing completeness\n3. Evaluating clarity\n4. Comparing helpfulness\nProvide detailed reasoning."
# }
def extract_plan_grpo_log_data(
    batch: BatchedDataDict[DatumSpec],
    stage1_repeated_batch: BatchedDataDict[DatumSpec],
    stage1_rewards: torch.Tensor,
    stage2_repeated_batch: BatchedDataDict[DatumSpec],
    stage2_rewards: torch.Tensor,
    num_plans_per_prompt: int,
    num_judgments_per_plan: int,
) -> Dict[str, Any]:
    """Extract comprehensive logging data for plan GRPO training.
    
    Returns a dictionary with stage1 plans and stage2 judgments for each original prompt.
    """
    num_original_prompts = len(batch["message_log"])
    
    # Helper to extract prompt and response from message log
    def extract_prompt_response(message_log):
        prompt = None
        response = None
        for message in message_log:
            if message["role"] == "user" and prompt is None:
                prompt = message["content"]
            elif message["role"] == "assistant":
                response = message["content"]
        return prompt or "", response or ""
    
    # Extract stage 1 data (first plan for each original prompt)
    stage1_prompts = []
    stage1_plans = []
    stage1_rewards_first = []
    
    for i in range(num_original_prompts):
        first_gen_idx = i * num_plans_per_prompt
        prompt, plan = extract_prompt_response(stage1_repeated_batch["message_log"][first_gen_idx])
        stage1_prompts.append(prompt)
        stage1_plans.append(plan)
        stage1_rewards_first.append(stage1_rewards[first_gen_idx].item())
    
    # Extract stage 2 data (first judgment for each original prompt)
    stage2_prompts = []
    stage2_judgments = []
    stage2_rewards_first = []
    
    for i in range(num_original_prompts):
        if i * num_plans_per_prompt * num_judgments_per_plan < len(stage2_repeated_batch["message_log"]):
            first_gen_idx = i * num_plans_per_prompt * num_judgments_per_plan
            prompt, judgment = extract_prompt_response(stage2_repeated_batch["message_log"][first_gen_idx])
            stage2_prompts.append(prompt)
            stage2_judgments.append(judgment)
            stage2_rewards_first.append(stage2_rewards[first_gen_idx].item())
        else:
            stage2_prompts.append("N/A")
            stage2_judgments.append("N/A")
            stage2_rewards_first.append(0.0)
    
    # Create log data with plan GRPO fields
    log_data = {
        "stage1_prompt": stage1_prompts,
        "stage1_plan": stage1_plans,
        "stage1_reward": stage1_rewards_first,
        "stage2_prompt": stage2_prompts,
        "stage2_judgment": stage2_judgments,
        "stage2_reward": stage2_rewards_first,
    }
    
    return log_data


def create_plan_prompts(
    original_batch: BatchedDataDict[DatumSpec],
    plan_prompt_template: str,
    tokenizer,
) -> BatchedDataDict[DatumSpec]:
    """Create prompts for generating evaluation plans/checklists/principles.
    
    Args:
        original_batch: Original batch containing the prompts
        plan_prompt_template: Template string for formatting plan generation prompts
        tokenizer: Tokenizer to use for tokenizing the prompts
    
    Returns:
        New batch with plan generation prompts
    """
    plan_batch = deepcopy(original_batch)
    
    # Create new message logs for plan generation
    new_message_logs = []
    
    for i, message_log in enumerate(original_batch["message_log"]):
        # Extract the original conversation_history from metadata
        if "extra_env_info" not in original_batch or not original_batch["extra_env_info"][i]:
            raise ValueError(f"No extra_env_info found in batch for sample {i}")
        
        conversation_history = original_batch["extra_env_info"][i].get("conversation_history")
        if conversation_history is None:
            raise ValueError(f"No conversation_history found in metadata for sample {i}")
        
        # Create the plan generation prompt using the template
        plan_prompt = plan_prompt_template.format(
            conversation_history=conversation_history
        )
        
        # Create a proper message structure for chat template
        plan_message = [
            {
                "role": "user",
                "content": plan_prompt,
            }
        ]
        
        # Apply chat template to get properly formatted content and token_ids
        formatted_content = tokenizer.apply_chat_template(
            plan_message,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        token_ids = tokenizer.apply_chat_template(
            plan_message,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=False,
            return_tensors="pt",
        )[0]
        
        # Create new message log with the plan generation prompt
        new_message_log = [
            {
                "role": "user",
                "content": formatted_content,
                "token_ids": token_ids,
            }
        ]
        new_message_logs.append(new_message_log)
    
    # Update plan batch with new data
    plan_batch["message_log"] = new_message_logs
    
    return plan_batch


def create_judgment_prompts(
    original_batch: BatchedDataDict[DatumSpec],
    plans: List[List[str]],
    judgment_prompt_template: str,
    tokenizer,
) -> BatchedDataDict[DatumSpec]:
    """Create prompts for generating judgments using the plans.
    
    Args:
        original_batch: Original batch containing the prompts
        plans: List of lists, where each inner list contains the plans for a prompt
        judgment_prompt_template: Template string for formatting judgment prompts
        tokenizer: Tokenizer to use for tokenizing the prompts
    
    Returns:
        New batch with judgment prompts
    """
    # Create new message logs for judgment
    new_message_logs = []
    new_extra_env_info = []
    new_loss_multiplier = []
    new_task_name = []
    
    for i, message_log in enumerate(original_batch["message_log"]):
        # Get the original question and metadata from extra_env_info
        if "extra_env_info" not in original_batch or not original_batch["extra_env_info"][i]:
            raise ValueError(f"No extra_env_info found in batch for sample {i}")
        
        original_metadata = original_batch["extra_env_info"][i]
        
        # For each plan, create a judgment prompt
        for plan_idx, plan in enumerate(plans[i]):
            # Create the judgment prompt using the template
            judgment_prompt = judgment_prompt_template.format(
                plan=plan,
                conversation_history=original_metadata.get("conversation_history", ""),
                response_1=original_metadata.get("response_1", ""),
                response_2=original_metadata.get("response_2", "")
            )
            
            # Create a proper message structure for chat template
            judgment_message = [
                {
                    "role": "user",
                    "content": judgment_prompt,
                }
            ]
            
            # Apply chat template to get properly formatted content and token_ids
            formatted_content = tokenizer.apply_chat_template(
                judgment_message,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )
            token_ids = tokenizer.apply_chat_template(
                judgment_message,
                tokenize=True,
                add_generation_prompt=True,
                add_special_tokens=False,
                return_tensors="pt",
            )[0]
            
            # Create new message log with the judgment prompt
            new_message_log = [
                {
                    "role": "user",
                    "content": formatted_content,
                    "token_ids": token_ids,
                }
            ]
            new_message_logs.append(new_message_log)
        
            # Keep track of corresponding metadata
            new_extra_env_info.append(original_metadata)
            new_loss_multiplier.append(original_batch["loss_multiplier"][i])
            new_task_name.append(original_batch["task_name"][i])
    
    # Ensure we have at least some valid judgment prompts
    if len(new_message_logs) == 0:
        raise ValueError("No valid judgment prompts could be created from the plans")
    
    print(f"  ✓ Created {len(new_message_logs)} judgment prompts from {len(original_batch['message_log'])} original prompts")
    
    # Create judgment batch with new data
    judgment_batch = {
        "message_log": new_message_logs,
        "extra_env_info": new_extra_env_info,
        "loss_multiplier": torch.tensor(new_loss_multiplier),
        "task_name": new_task_name,
    }
    
    return BatchedDataDict[DatumSpec](judgment_batch)


def aggregate_judgment_rewards_to_plans(
    stage1_rewards: torch.Tensor,
    stage2_rewards: torch.Tensor,
    num_prompts: int,
    num_plans_per_prompt: int,
    num_judgments_per_plan: int,
) -> torch.Tensor:
    """Aggregate judgment rewards to compute plan rewards.
    
    Each plan's reward is the average reward of all judgments that used that plan.
    
    Args:
        stage1_rewards: Placeholder rewards for plans (will be replaced)
        stage2_rewards: Actual rewards from judgments
        num_prompts: Number of original prompts
        num_plans_per_prompt: Number of plans per prompt
        num_judgments_per_plan: Number of judgments per plan
    
    Returns:
        Updated plan rewards based on judgment averages
    """
    # Reshape stage2 rewards to (num_prompts, num_plans_per_prompt, num_judgments_per_plan)
    stage2_rewards_reshaped = stage2_rewards.view(num_prompts, num_plans_per_prompt, num_judgments_per_plan)
    
    # Average across judgments to get plan rewards
    plan_rewards = stage2_rewards_reshaped.mean(dim=2)
    
    # Flatten back to match original shape
    plan_rewards_flat = plan_rewards.view(-1)
    
    return plan_rewards_flat


# ===============================================================================
# Type Definitions
# ===============================================================================
class PlanGRPOSaveState(TypedDict):
    """State that needs to be saved/loaded for plan GRPO training."""
    step: int
    optim_step: int
    consumed_samples: int
    num_epochs: int


class MasterConfig(TypedDict):
    """Master configuration for plan GRPO training."""
    tokenizer: Dict[str, Any]
    generation: Dict[str, Any]
    policy: PolicyConfig
    cluster: ClusterConfig
    data: DataConfig
    val_data: Optional[DataConfig]
    env: Dict[str, Any]
    val_env: Optional[Dict[str, Any]]
    loss_fn: ClippedPGLossConfig
    plan_grpo: Dict[str, Any]
    logger: LoggerConfig
    checkpointing: CheckpointingConfig


# ===============================================================================
# Core Functions
# ===============================================================================
def apply_environment_post_processing(
    batch: BatchedDataDict[DatumSpec],
    task_to_env: Dict[str, EnvironmentInterface],
    prefix: str = "",
) -> Tuple[BatchedDataDict[DatumSpec], Dict[str, float]]:
    """Apply environment-specific post-processing to a batch.
    
    Returns updated batch and metrics dictionary.
    """
    # Group samples by task
    task_groups = defaultdict(list)
    for i, task in enumerate(batch["task_name"]):
        task_groups[task].append(i)

    # Process each task group
    all_returns = []
    for task, indices in task_groups.items():
        if task not in task_to_env:
            print(f"⚠️  Warning: No environment found for task '{task}', skipping post-processing")
            # Create dummy returns
            for idx in indices:
                all_returns.append({
                    "total_reward": batch.get("total_reward", torch.zeros(len(batch["task_name"])))[idx].item()
                    if "total_reward" in batch else 0.0,
                })
            continue

        # Create sub-batch for this task
        task_batch = batch.filter(indices)
        
        # Apply environment post-processing
        env_result = task_to_env[task].post_process_batch(task_batch)
        
        # Store results
        for i, idx in enumerate(indices):
            all_returns.append({
                "total_reward": env_result.rewards[i].item(),
                **env_result.extra_info[i] if i < len(env_result.extra_info) else {}
            })

    # Update batch with results
    batch["total_reward"] = torch.tensor([r["total_reward"] for r in all_returns], dtype=torch.float32)
    
    # Collect metrics
    metrics = {}
    reward_array = batch["total_reward"].numpy()
    metrics[f"{prefix}reward/mean"] = float(np.mean(reward_array))
    metrics[f"{prefix}reward/std"] = float(np.std(reward_array))
    metrics[f"{prefix}reward/min"] = float(np.min(reward_array))
    metrics[f"{prefix}reward/max"] = float(np.max(reward_array))
    
    return batch, metrics


class TimeLimitTimer:
    """Timer to track time limits during training."""
    def __init__(self, time_limit: Optional[str]):
        self.time_limit = time_limit
        self._start_time = None
        
    def start_time(self):
        self._start_time = time.time()
        
    def is_finished(self) -> bool:
        if self.time_limit is None or self._start_time is None:
            return False
        limit_seconds = pd.Timedelta(self.time_limit).total_seconds()
        return (time.time() - self._start_time) >= limit_seconds


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer,
    val_task_to_env: Dict[str, EnvironmentInterface],
    step: int,
    master_config: MasterConfig,
    logger: Logger,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Run validation for plan GRPO.
    
    Returns validation metrics and timing information.
    """
    timer = Timer()
    all_metrics = {}
    
    print(f"\n{'='*60}")
    print(f"Validation at step {step}")
    print(f"{'='*60}")
    
    # For plan GRPO, we would need to:
    # 1. Generate plans for validation prompts
    # 2. Generate judgments using those plans
    # 3. Calculate metrics
    
    # For now, return empty metrics (to be implemented based on specific needs)
    return all_metrics, timer.get_all_timings()


# ===============================================================================
# Setup Function
# ===============================================================================
def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    dataset,
    val_dataset=None,
) -> Tuple[
    PolicyInterface,
    Optional[GenerationInterface],
    RayVirtualCluster,
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    LossFunction,
    Logger,
    CheckpointManager,
    PlanGRPOSaveState,
    MasterConfig,
]:
    """Setup all components needed for plan GRPO training."""
    print("\n" + "=" * 60)
    print(" " * 20 + "PLAN GRPO SETUP")
    print("=" * 60 + "\n")



    # ==========================
    #   Config Extraction
    # ==========================
    generation_config = master_config["generation"]
    policy_config = master_config["policy"]
    cluster_config = master_config["cluster"]
    data_config = master_config["data"]
    val_data_config = master_config.get("val_data")
    loss_config = master_config["loss_fn"]
    logger_config = master_config["logger"]
    checkpointing_config = master_config["checkpointing"]
    plan_grpo_config = master_config["plan_grpo"]

    # ==========================
    #   Checkpoint Management
    # ==========================
    print("\n▶ Setting up checkpointing...")
    checkpointer = CheckpointManager(config=checkpointing_config)
    last_checkpoint_path = checkpointer.find_last_checkpoint()
    if last_checkpoint_path:
        print(f"  ✓ Found checkpoint: {last_checkpoint_path}")
        plan_grpo_save_state = checkpointer.load_state(last_checkpoint_path / "plan_grpo_state.pkl")
        print(f"    • Resuming from step: {plan_grpo_save_state['step']}")
        print(f"    • Consumed samples: {plan_grpo_save_state['consumed_samples']}")
    else:
        print("  ✓ No checkpoint found, starting fresh")
        plan_grpo_save_state = PlanGRPOSaveState(
            step=0,
            optim_step=0,
            consumed_samples=0,
            num_epochs=plan_grpo_config["num_epochs"],
        )

    # ==========================
    #   Logger
    # ==========================
    print("\n▶ Setting up logger...")
    logger = Logger(config=logger_config)
    print(f"  ✓ Logger initialized")

    # ==========================
    #   Datasets
    # ==========================
    print("\n▶ Setting up datasets...")

    # Training dataset
    print(f"  ✓ Training dataset loaded with {len(dataset)} examples")
    
    # Setup data generators for shuffling
    shuffle_train = data_config["train"]["shuffle"]
    shuffle_val = data_config["val"]["shuffle"] if "val" in data_config else False
    
    train_data_generator = None
    val_data_generator = None
    
    if shuffle_train:
        train_data_generator = torch.Generator()
        train_data_generator.manual_seed(data_config["train"]["seed"])
    
    if shuffle_val and val_dataset:
        val_data_generator = torch.Generator()
        val_data_generator.manual_seed(data_config["val"]["seed"])

    dataloader = StatefulDataLoader(
        dataset,
        batch_size=plan_grpo_config["num_prompts_per_step"],
        shuffle=shuffle_train,
        generator=train_data_generator,
        collate_fn=rl_collate_fn,
        drop_last=data_config["train"]["drop_last"],
    )
    
    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            Path(last_checkpoint_path) / "train_dataloader.pt"
        )
        dataloader.load_state_dict(dataloader_state_dict)
    
    print(f"    • Batch size: {plan_grpo_config['num_prompts_per_step']}")
    print(f"    • Steps per epoch: {len(dataloader)}")

    # Validation dataset
    val_dataloader = None
    # If validation is enabled, load the validation dataloader
    if plan_grpo_config["val_period"] > 0 or plan_grpo_config["val_at_start"]:
        if val_dataset:
            val_batch_size = min(plan_grpo_config["max_val_samples"], len(val_dataset))
            if "val_batch_size" in plan_grpo_config:
                val_batch_size = plan_grpo_config["val_batch_size"]
            
            val_dataloader = StatefulDataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=shuffle_val,
                generator=val_data_generator,
                collate_fn=rl_collate_fn,
                drop_last=val_data_config["drop_last"] if val_data_config else False,
            )
            print(f"  ✓ Validation dataset loaded with {len(val_dataset)} examples")
        else:
            print("  ⚠️ Validation requested but no validation dataset provided")
    else:
        print("  ℹ Validation disabled")



    # ==========================
    #   Ray Cluster
    # ==========================
    print("\n▶ Setting up Ray cluster...")
    colocated_inference = generation_config.get("max_colocated_worker_groups", 1) > 1
    cluster = RayVirtualCluster(
        name="plan_grpo_policy_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=2 if colocated_inference else 1,
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # ==========================
    #   Training and Inference
    # ==========================
    print("\n▶ Setting up model and training...")

    # vllm model loading prefers clean environment, initialize policy_generation before policy
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    if backend == "hf":
        policy_generation = None
        print(f"  ✓ Using HF backend for generation with {policy_config['model_name']}")
    elif backend == "vllm":
        policy_generation = VllmGeneration(cluster=cluster, config=generation_config)
        # Worker groups are not initialized until the first call to run something on workergroups.
        policy_generation.finish_generation()
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}"
        )

    # Setup tokenizer if needed for HfPolicy
    if 'tokenizer' in policy_config and 'name' in policy_config['tokenizer']:
        from nemo_rl.algorithms.utils import get_tokenizer
        policy_tokenizer = get_tokenizer(policy_config['tokenizer'])
    else:
        policy_tokenizer = tokenizer
        
    policy = HfPolicy(
        cluster=cluster,
        config=policy_config,
        tokenizer=policy_tokenizer,
        weights_path=Path(last_checkpoint_path) / "policy" / "weights"
        if last_checkpoint_path
        else None,
        optimizer_path=Path(last_checkpoint_path) / "policy" / "optimizer"
        if last_checkpoint_path
        else None,
        init_optimizer=True,
    )

    loss_fn = ClippedPGLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        plan_grpo_save_state,
        master_config,
    )


def get_reasoning_split_word(env_configs: Dict[str, Any]) -> Optional[str]:
    """Get reasoning_split_word from any enabled environment."""
    for env_name, env_config in env_configs.items():
        if env_config.get("enable", False) and "reasoning_split_word" in env_config:
            return env_config["reasoning_split_word"]
    return None


def refit_policy_generation(
    policy: PolicyInterface,
    policy_generation: GenerationInterface,
    refit_buffer_size_gb: int,  # GB
):
    """Refit the policy generation interface with the latest policy weights."""
    policy.offload_before_refit()
    policy_generation.prepare_for_generation(tags=["weights"])
    # Streaming update weights to save memory
    state_dict_info = policy.prepare_weights_for_ipc()
    # group keys to save time
    available_bytes = refit_buffer_size_gb * (1024**3)
    split_keys, keys = [], []
    for key, size_in_bytes in state_dict_info:
        if size_in_bytes > available_bytes:
            if keys:
                split_keys.append(keys)
                keys = []
            available_bytes = refit_buffer_size_gb * (1024**3)

        keys.append(key)
        available_bytes -= size_in_bytes

    if len(keys) > 0:
        split_keys.append(keys)
    # do update
    for keys in split_keys:
        ipc_handles = policy.get_weights_ipc_handles(keys)
        if not policy_generation.update_weights(ipc_handles):
            error_message = (
                "❌ Error: Updating weights for the generation policy failed during refit.\n"
                "This often indicates an issue with cuda-ipc or "
                "a problem within the generation backend (e.g., vLLM worker).\n"
            )
            raise RuntimeError(error_message)
    policy.offload_after_refit()
    policy_generation.prepare_for_generation(tags=["kv_cache"])


# ===============================================================================
# Training & Validation
# ===============================================================================
def plan_grpo_train(
    policy: PolicyInterface,
    policy_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    loss_fn: LossFunction,
    task_to_env: Dict[str, EnvironmentInterface],
    val_task_to_env: Optional[Dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    plan_grpo_save_state: Optional[PlanGRPOSaveState],
    master_config: MasterConfig,
):
    """Run Plan GRPO training algorithm."""
    timer = Timer()
    NEED_REFIT = True
    # If policy_generation is None, use the policy as the generation interface (hf framework backend)
    if policy_generation is None:
        policy_generation = policy
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running

    # common config/state items
    step = plan_grpo_save_state["step"]
    optim_step = plan_grpo_save_state["optim_step"]

    consumed_samples = plan_grpo_save_state["consumed_samples"]
    val_period = master_config["plan_grpo"]["val_period"]
    val_at_start = master_config["plan_grpo"]["val_at_start"]
    refit_buffer_size_gb = master_config["policy"]["refit_buffer_size_gb"]

    num_epochs = master_config["plan_grpo"]["num_epochs"]
    max_num_steps = num_epochs * len(dataloader)

    # Initialize time limit timer
    time_limit_timer = TimeLimitTimer(master_config["plan_grpo"].get("time_limit"))
    time_limit_timer.start_time()

    # Run validation at the start if configured
    if val_at_start and step == 0:
        print("\n🔍 Running initial validation...")
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, policy_generation, refit_buffer_size_gb)
            POLICY_GENERATION_STALE = False
        else:
            policy_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=0,
            master_config=master_config,
            logger=logger,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, step, prefix="validation")
        logger.log_metrics(validation_timings, step, prefix="timing/validation")

    # Run plan GRPO training
    batch: BatchedDataDict[DatumSpec]
    iter_dataloader = iter(dataloader)

    while step < max_num_steps and not time_limit_timer.is_finished():
        try:
            batch = next(iter_dataloader)
        except StopIteration:
            iter_dataloader = iter(dataloader)
            batch = next(iter_dataloader)

        print(f"\n{'=' * 25} Step {step + 1}/{max_num_steps} {'=' * 25}")
        val_metrics, validation_timings = None, None

        with timer.time("total_step_time"):
            # Check if we should skip stage 1
            skip_stage1 = master_config["plan_grpo"].get("skip_stage1", False)
            use_baseline_plan = master_config["plan_grpo"].get("use_baseline_plan", False)
            
            if skip_stage1 and not use_baseline_plan:
                raise ValueError("skip_stage1 requires use_baseline_plan to be True")
            
            # ============== Stage 1: Plan Generation ==============
            if not skip_stage1:
                print("\n▶ Stage 1: Plan Generation...")
            else:
                print("\n▶ Stage 1: Skipped (using baseline plan only)")
            
            # Initialize variables that will be used later
            stage1_repeated_batch = None
            stage1_rollout_metrics = {}
            stage1_input_ids = None
            
            if not skip_stage1:
                with timer.time("stage1_preparation"):
                    # Create plan generation prompts
                    plan_batch = create_plan_prompts(
                        batch,
                        master_config["plan_grpo"]["plan_prompt_template"],
                        tokenizer,
                    )
                    
                    # Repeat batch items for multiple plan generations
                    stage1_repeated_batch: BatchedDataDict[DatumSpec] = plan_batch.repeat_interleave(
                        master_config["plan_grpo"]["num_plans_per_prompt"]
                    )
                    # Convert LLMMessageLogType to FlatMessagesType for generation and save prompt ids
                    batched_flat_pre_gen, _ = batched_message_log_to_flat_message(
                        stage1_repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    )
                    stage1_input_ids = batched_flat_pre_gen["token_ids"]  # prompt-only ids (no assistant)

                # Generate plans for stage 1
                print(f"  • Generating {stage1_repeated_batch.size} evaluation plans...")
                with timer.time("stage1_generation_prep"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            policy,
                            policy_generation,
                            refit_buffer_size_gb,
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        policy_generation.prepare_for_generation()

                with timer.time("stage1_generation"):
                    stage1_repeated_batch, stage1_rollout_metrics = run_multi_turn_rollout(
                        policy_generation=policy_generation,
                        input_batch=stage1_repeated_batch,
                        tokenizer=tokenizer,
                        task_to_env=task_to_env,
                        max_seq_len=master_config["policy"]["max_total_sequence_length"],
                        max_rollout_turns=master_config["plan_grpo"]["max_rollout_turns"],
                        greedy=False,
                    )
                    
                    # Keep generation active for stage 2

                # Apply environment post-processing for stage 1 (plans)
                print("  • Applying stage 1 environment post-processing...")
                with timer.time("stage1_env_post_processing"):
                    if master_config["plan_grpo"].get("skip_stage1_env_post_processing", False):
                        stage1_env_metrics = {}
                        print("    (Skipped - skip_stage1_env_post_processing is enabled)")
                    else:
                        stage1_repeated_batch, stage1_env_metrics = apply_environment_post_processing(
                            stage1_repeated_batch, task_to_env, prefix="stage1_"
                        )
                    stage1_rollout_metrics.update(stage1_env_metrics)
            else:
                # When skipping stage1, we need to prepare for generation here
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(
                        policy,
                        policy_generation,
                        refit_buffer_size_gb,
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()

            # Extract stage 1 plans
            num_prompts = len(batch["message_log"])
            num_plans = master_config["plan_grpo"]["num_plans_per_prompt"]
            
            # Get reasoning split word from any enabled environment
            reasoning_split_word = get_reasoning_split_word(master_config["env"])
            
            stage1_plans = []
            json_parse_failures = []  # Track which plans failed JSON parsing
            
            if not skip_stage1:
                # Group plans by original prompt and track JSON parsing failures
                for i in range(num_prompts):
                    prompt_plans = []
                    for j in range(num_plans):
                        idx = i * num_plans + j
                        # Extract assistant response (plan) from the message log
                        last_assistant_response = None
                        for message in stage1_repeated_batch["message_log"][idx]:
                            if message["role"] == "assistant":
                                last_assistant_response = message["content"]
                        if last_assistant_response is not None:
                            if reasoning_split_word and reasoning_split_word in last_assistant_response:
                                # Remove reasoning part if split word exists
                                last_assistant_response = last_assistant_response.split(reasoning_split_word)[-1].lstrip()
                            
                            # Try to parse JSON response to extract plan
                            try:
                                json_response = json.loads(last_assistant_response)
                                if isinstance(json_response, dict) and "plan" in json_response:
                                    prompt_plans.append(json_response["plan"])
                                    json_parse_failures.append(False)
                                else:
                                    # Fallback to simple default plan if not proper JSON format
                                    default_plan = "First generate your own response, compare with the provided response and judge them."
                                    prompt_plans.append(default_plan)
                                    json_parse_failures.append(True)
                                    print(f"  ⚠️ Plan {idx}: Invalid JSON format (missing 'plan' key) - using default plan")
                            except (json.JSONDecodeError, ValueError) as e:
                                # Fallback to simple default plan if JSON parsing fails
                                default_plan = "First generate your own response, compare with the provided response and judge them."
                                prompt_plans.append(default_plan)
                                json_parse_failures.append(True)
                                print(f"  ⚠️ Plan {idx}: JSON parsing failed - {str(e)[:50]}... - using default plan")
                        else:
                            # Use default plan if no assistant response found
                            default_plan = "First generate your own response, compare with the provided response and judge them."
                            prompt_plans.append(default_plan)
                            json_parse_failures.append(True)
                            print(f"  ⚠️ Plan {idx}: No assistant response found - using default plan")
                    
                    stage1_plans.append(prompt_plans)
            else:
                # When skipping stage1, create empty plan lists that will be filled with baseline plan
                for i in range(num_prompts):
                    stage1_plans.append([])  # Empty list that will have baseline plan added in stage 2
            
            # ============== Stage 2: Judgments using Plans ==============
            print("\n▶ Stage 2: Judgments using Plans...")
            with timer.time("stage2_preparation"):
                # If using baseline plan, add it to the plans for stage 2
                stage2_plans = stage1_plans
                if use_baseline_plan:
                    baseline_plan = master_config["plan_grpo"].get(
                        "baseline_plan_template",
                        "Carefully evaluate both responses by checking accuracy, completeness, and helpfulness. Compare them fairly and provide detailed reasoning."
                    )
                    if skip_stage1:
                        # When skipping stage1, only use the baseline plan
                        stage2_plans = [[baseline_plan] for _ in range(num_prompts)]
                    else:
                        # Create a copy and add baseline plan to each prompt's plans
                        stage2_plans = [plans + [baseline_plan] for plans in stage1_plans]
                
                # Create judgment prompts using the plans
                judgment_batch = create_judgment_prompts(
                    batch,
                    stage2_plans,
                    master_config["plan_grpo"]["judgment_prompt_template"],
                    tokenizer,
                )
                
                # Repeat each judgment prompt for multiple generations per plan
                stage2_repeated_batch = judgment_batch.repeat_interleave(
                    master_config["plan_grpo"]["num_judgments_per_plan"]
                )
                
                # Calculate input_ids for stage 2
                stage2_flat_pre_rollout, _ = batched_message_log_to_flat_message(
                    stage2_repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                )
                stage2_input_ids = stage2_flat_pre_rollout["token_ids"]

            print(f"  • Generating {stage2_repeated_batch.size} judgments...")
            with timer.time("stage2_generation"):
                stage2_repeated_batch, stage2_rollout_metrics = run_multi_turn_rollout(
                    policy_generation=policy_generation,
                    input_batch=stage2_repeated_batch,
                    tokenizer=tokenizer,
                    task_to_env=task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["plan_grpo"]["max_rollout_turns"],
                    greedy=False,
                )
                
                policy_generation.finish_generation()

            # Apply environment post-processing for stage 2 (judgments)
            print("  • Applying stage 2 environment post-processing...")
            with timer.time("stage2_env_post_processing"):
                if master_config["plan_grpo"].get("skip_stage2_env_post_processing", False):
                    stage2_env_metrics = {}
                    print("    (Skipped - skip_stage2_env_post_processing is enabled)")
                else:
                    stage2_repeated_batch, stage2_env_metrics = apply_environment_post_processing(
                        stage2_repeated_batch, task_to_env, prefix="stage2_"
                    )
                stage2_rollout_metrics.update(stage2_env_metrics)

            # Extract stage 2 rewards (judgment rewards)
            stage2_rewards = stage2_repeated_batch["total_reward"]

            # ============== Calculate Plan Rewards ==============
            print("\n▶ Calculating plan rewards from judgment averages...")
            with timer.time("reward_calculation"):
                # Aggregate judgment rewards to get plan rewards
                # Note: num_plans_to_aggregate includes baseline plan if enabled
                if skip_stage1:
                    # When skipping stage1, we only have 1 plan per prompt (baseline)
                    num_plans_to_aggregate = 1
                else:
                    num_plans_to_aggregate = master_config["plan_grpo"]["num_plans_per_prompt"]
                    if use_baseline_plan:
                        num_plans_to_aggregate += 1
                    
                stage1_rewards = aggregate_judgment_rewards_to_plans(
                    stage1_rewards=None,  # Not used, will be replaced
                    stage2_rewards=stage2_rewards,
                    num_prompts=num_prompts,
                    num_plans_per_prompt=num_plans_to_aggregate,
                    num_judgments_per_plan=master_config["plan_grpo"]["num_judgments_per_plan"],
                )
                
                if skip_stage1:
                    # When skipping stage1, we don't have any generated plans to process
                    stage1_generated_plan_rewards = torch.empty(0)  # Empty tensor
                    stage1_all_plan_rewards = stage1_rewards  # Only baseline plan rewards
                    # No JSON parsing failures when skipping stage1
                    json_parse_failures_tensor = torch.zeros(0, dtype=torch.bool)
                else:
                    # If using baseline plan, we need to handle the fact that stage1 only has 8 plans
                    # but stage2 has 9 plans (8 generated + 1 baseline)
                    if use_baseline_plan:
                        # Extract rewards for generated plans only (first 8)
                        stage1_generated_plan_rewards = stage1_rewards.view(num_prompts, num_plans_to_aggregate)[:, :num_plans].reshape(-1)
                        
                        # Apply penalty for plans that failed JSON parsing (only for generated plans)
                        json_parse_failures_generated = json_parse_failures[:len(stage1_generated_plan_rewards)]
                        json_parse_failures_tensor = torch.tensor(json_parse_failures_generated, dtype=torch.bool)
                        penalty_reward = master_config["plan_grpo"].get("json_parse_failure_penalty", -50.0)
                        stage1_generated_plan_rewards[json_parse_failures_tensor] = penalty_reward
                        
                        # Update stage1 batch with rewards (only for generated plans)
                        stage1_repeated_batch["total_reward"] = stage1_generated_plan_rewards
                        
                        # Keep full rewards including baseline for later use
                        stage1_all_plan_rewards = stage1_rewards
                    else:
                        # Apply penalty for plans that failed JSON parsing
                        json_parse_failures_tensor = torch.tensor(json_parse_failures, dtype=torch.bool)
                        penalty_reward = master_config["plan_grpo"].get("json_parse_failure_penalty", -50.0)
                        stage1_rewards[json_parse_failures_tensor] = penalty_reward
                        
                        # Update stage1 batch with calculated rewards
                        stage1_repeated_batch["total_reward"] = stage1_rewards
                        stage1_all_plan_rewards = stage1_rewards
                        stage1_generated_plan_rewards = stage1_rewards
                
                # Log statistics about JSON parsing failures
                num_failures = json_parse_failures_tensor.sum().item()
                if num_failures > 0:
                    print(f"  ⚠️ Applied {penalty_reward} penalty to {num_failures}/{len(json_parse_failures_tensor)} plans that failed JSON parsing")
                
                # Calculate advantages for plans
                print("  • Computing plan advantages...")
                
                if skip_stage1:
                    # When skipping stage1, we don't calculate advantages for plans
                    # as there are no generated plans to train on
                    stage1_advantages = torch.empty(0)
                    print("    (No plan advantages - training only on judgments)")
                else:
                    # Extract baseline plan rewards if using baseline plan
                    baseline_plan_rewards = None
                    if use_baseline_plan:
                        # Baseline plan is the last plan for each prompt in the full rewards
                        baseline_indices = [(i+1) * num_plans_to_aggregate - 1 for i in range(num_prompts)]
                        baseline_plan_rewards = stage1_all_plan_rewards[baseline_indices]
                        
                        # For GRPO baseline calculation, use only the generated plans
                        stage1_rewards_for_grpo = stage1_generated_plan_rewards
                        stage1_input_ids_for_grpo = stage1_input_ids
                    else:
                        stage1_rewards_for_grpo = stage1_repeated_batch["total_reward"]
                        stage1_input_ids_for_grpo = stage1_input_ids
                    
                    # Check for potentially problematic configuration
                    expected_responses = master_config["plan_grpo"]["num_plans_per_prompt"] if master_config["plan_grpo"].get("check_baseline_correctness", True) else None
                    
                    # Get prompts for baseline calculation (use stage1 input ids)
                    stage1_grpo_baseline, stage1_grpo_std, stage1_grpo_metrics = calculate_baseline_and_std_per_prompt(
                        prompts=stage1_input_ids_for_grpo,
                        rewards=stage1_rewards_for_grpo,
                        valid_mask=torch.ones_like(stage1_rewards_for_grpo),
                        leave_one_out_baseline=master_config["plan_grpo"]["use_leave_one_out_baseline"],
                        expected_responses_per_prompt=expected_responses,
                    )
                    stage1_rollout_metrics.update(stage1_grpo_metrics)
                
                    # Apply max with baseline plan rewards if enabled
                    if use_baseline_plan:
                        # Apply max operation: baseline = max(GRPO_baseline, baseline_plan_reward)
                        baseline_plan_rewards_expanded = baseline_plan_rewards.repeat_interleave(num_plans)
                        stage1_final_baseline = torch.maximum(stage1_grpo_baseline, baseline_plan_rewards_expanded)
                        
                        # Log baseline plan effectiveness
                        baseline_improvement = (baseline_plan_rewards_expanded > stage1_grpo_baseline).float().mean()
                        stage1_rollout_metrics["baseline_plan_improvement_ratio"] = float(baseline_improvement)
                        stage1_rollout_metrics["baseline_plan_reward_mean"] = float(baseline_plan_rewards.mean())
                    else:
                        stage1_final_baseline = stage1_grpo_baseline
                    
                    # Calculate advantages only for generated plans
                    stage1_advantages = (stage1_generated_plan_rewards - stage1_final_baseline).unsqueeze(-1)
                    
                    if master_config["plan_grpo"]["normalize_rewards"]:
                        # Don't sharpen the ones with no variation
                        zero_std_mask = stage1_grpo_std > 0
                        stage1_advantages[zero_std_mask] = (
                            stage1_advantages[zero_std_mask] / stage1_grpo_std.unsqueeze(-1)[zero_std_mask]
                        )
                
                # Calculate advantages for judgments
                print("  • Computing judgment advantages...")
                
                # Check for potentially problematic configuration
                expected_responses_stage2 = master_config["plan_grpo"]["num_judgments_per_plan"] if master_config["plan_grpo"].get("check_baseline_correctness", True) else None
                
                # Get prompts for baseline calculation (use stage2 input ids)
                stage2_baseline, stage2_std, stage2_more_metrics = calculate_baseline_and_std_per_prompt(
                    prompts=stage2_input_ids,
                    rewards=stage2_rewards,
                    valid_mask=torch.ones_like(stage2_rewards),
                    leave_one_out_baseline=master_config["plan_grpo"]["use_leave_one_out_baseline"],
                    expected_responses_per_prompt=expected_responses_stage2,
                )
                stage2_rollout_metrics.update(stage2_more_metrics)
                stage2_advantages = (stage2_rewards - stage2_baseline).unsqueeze(-1)
                
                if master_config["plan_grpo"]["normalize_rewards"]:
                    # Don't sharpen the ones with no variation
                    zero_std_mask = stage2_std > 0
                    stage2_advantages[zero_std_mask] = (
                        stage2_advantages[zero_std_mask] / stage2_std.unsqueeze(-1)[zero_std_mask]
                    )

            # ============== Prepare Training Data ==============
            print("\n▶ Preparing training data...")
            with timer.time("train_data_preparation"):
                # Filter out baseline plan judgments from stage2 if using baseline plan
                # BUT only when not skipping stage1 (otherwise baseline judgments are all we have!)
                if use_baseline_plan and not skip_stage1:
                    # Calculate which stage2 samples correspond to baseline plans
                    stage2_baseline_indices = []
                    for i in range(num_prompts):
                        # Baseline plan is the last plan for each prompt
                        baseline_plan_idx = i * num_plans_to_aggregate + num_plans
                        start_idx = baseline_plan_idx * master_config["plan_grpo"]["num_judgments_per_plan"]
                        end_idx = start_idx + master_config["plan_grpo"]["num_judgments_per_plan"]
                        stage2_baseline_indices.extend(range(start_idx, end_idx))
                    
                    stage2_mask = torch.ones(len(stage2_repeated_batch["message_log"]), dtype=torch.bool)
                    stage2_mask[stage2_baseline_indices] = False
                    
                    # Filter stage2 batch
                    stage2_indices_to_keep = torch.where(stage2_mask)[0].tolist()
                    stage2_repeated_batch = stage2_repeated_batch.filter(stage2_indices_to_keep)
                    stage2_advantages = stage2_advantages[stage2_mask]
                    
                    print(f"  • Filtered out {len(stage2_baseline_indices)} baseline plan judgments from training")
                
                # Update message logs with advantages for stage 1
                if not skip_stage1:
                    for i, message_log in enumerate(stage1_repeated_batch["message_log"]):
                        for message in message_log:
                            if message["role"] == "assistant":
                                if "generation_logprobs" not in message:
                                    message["generation_logprobs"] = torch.zeros_like(
                                        message["token_ids"], dtype=torch.float32
                                    )
                                message["advantages"] = stage1_advantages[i].expand(
                                    message["token_ids"].shape
                                )
                
                # Update message logs with advantages for stage 2
                for i, message_log in enumerate(stage2_repeated_batch["message_log"]):
                    for message in message_log:
                        if message["role"] == "assistant":
                            if "generation_logprobs" not in message:
                                message["generation_logprobs"] = torch.zeros_like(
                                    message["token_ids"], dtype=torch.float32
                                )
                            message["advantages"] = stage2_advantages[i].expand(
                                message["token_ids"].shape
                            )
                
                # Convert updated message logs to flat messages for training
                if not skip_stage1:
                    stage1_flat_messages, stage1_input_lengths = batched_message_log_to_flat_message(
                        stage1_repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )
                    
                    # Create training data from flattened messages
                    stage1_train_data = BatchedDataDict[ClippedPGLossDataDict]({
                        "input_ids": stage1_flat_messages["token_ids"],
                        "input_lengths": stage1_input_lengths,
                        "advantages": stage1_flat_messages["advantages"],
                        "generation_logprobs": stage1_flat_messages["generation_logprobs"],
                        "token_mask": stage1_flat_messages["token_loss_mask"],
                        "sample_mask": stage1_repeated_batch["loss_multiplier"],
                    })
                    stage1_train_data.to("cpu")
                
                stage2_flat_messages, stage2_input_lengths = batched_message_log_to_flat_message(
                    stage2_repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    make_sequence_length_divisible_by=master_config["policy"][
                        "make_sequence_length_divisible_by"
                    ],
                )
                
                stage2_train_data = BatchedDataDict[ClippedPGLossDataDict]({
                    "input_ids": stage2_flat_messages["token_ids"],
                    "input_lengths": stage2_input_lengths,
                    "advantages": stage2_flat_messages["advantages"],
                    "generation_logprobs": stage2_flat_messages["generation_logprobs"],
                    "token_mask": stage2_flat_messages["token_loss_mask"],
                    "sample_mask": stage2_repeated_batch["loss_multiplier"],
                })
                stage2_train_data.to("cpu")
                
                # Combine training data from both stages or use stage2 only
                if skip_stage1:
                    combined_train_data = stage2_train_data
                else:
                    combined_train_data = combine_training_data(
                        stage1_train_data, stage2_train_data
                    )

            # ============== Logprob Inference ==============
            print("\n▶ Preparing for logprob inference...")
            with timer.time("logprob_inference_prep"):
                policy.prepare_for_lp_inference()

            print("▶ Computing logprobs...")
            with timer.time("policy_and_reference_logprobs"):
                fprop_logprobs = policy.get_logprobs(combined_train_data)["logprobs"]
                reference_logprobs = policy.get_reference_policy_logprobs(combined_train_data)[
                    "reference_logprobs"
                ]
                combined_train_data["prev_logprobs"] = fprop_logprobs
                combined_train_data["reference_policy_logprobs"] = reference_logprobs

            # ============== Policy Update ==============
            print("\n▶ Preparing for training...")
            with timer.time("training_prep"):
                policy.prepare_for_training()  # set model train and reload optim to GPU
                POLICY_GENERATION_STALE = True

            print("▶ Training policy...")
            with timer.time("policy_training"):
                list_of_train_metrics = policy.train(combined_train_data, loss_fn)

            for i, m in enumerate(list_of_train_metrics):
                to_log = optim_step + i

                grad_sparsity_dict = m.pop("grad_sparsity_dict")
                log_dir = master_config["logger"]["log_dir"]
                sparsity_file_path = os.path.join(
                    log_dir, f"optim_step_{to_log}_grad_sparsity.json"
                )

                with open(sparsity_file_path, "w") as f:
                    json.dump(grad_sparsity_dict, f, indent=2)

                print(f"Saved grad sparsity to {sparsity_file_path}")

            optim_step += len(list_of_train_metrics)

            # ============== Logging ==============
            print("\n▶ Logging metrics...")
            with timer.time("logging"):
                # Log rollout metrics
                rollout_metrics = {
                    **stage1_rollout_metrics,
                    **stage2_rollout_metrics,
                    "stage1_reward_mean": float(stage1_rewards.mean()),
                    "stage1_reward_std": float(stage1_rewards.std()),
                    "stage2_reward_mean": float(stage2_rewards.mean()),
                    "stage2_reward_std": float(stage2_rewards.std()),
                    "stage1_baseline_mean": float(stage1_baseline.mean()),
                    "stage1_baseline_std": float(stage1_std.mean()),
                    "stage2_baseline_mean": float(stage2_baseline.mean()),
                    "stage2_baseline_std": float(stage2_std.mean()),
                }
                
                logger.log_metrics(rollout_metrics, step)
                logger.log_metrics(timer.get_all_timings(), step, prefix="timing")
                
                # Log training metrics for each optimization step
                for i, train_step_metric in enumerate(list_of_train_metrics):
                    train_step_metric["optim_step"] = optim_step - len(list_of_train_metrics) + i + 1
                    train_step_metric["outer_loop_step"] = step + 1
                    logger.log_metrics(
                        train_step_metric,
                        train_step_metric["optim_step"],
                        prefix="train_step"
                    )
                
                # Sample logging (not part of grpo config but useful for debugging)
                sample_log_period = master_config["plan_grpo"].get("sample_log_period", 10)
                if sample_log_period > 0 and step % sample_log_period == 0 and not skip_stage1:
                    log_data = extract_plan_grpo_log_data(
                        batch,
                        stage1_repeated_batch,
                        stage1_generated_plan_rewards,  # Use the rewards for generated plans only
                        stage2_repeated_batch,
                        stage2_rewards,
                        master_config["plan_grpo"]["num_plans_per_prompt"],
                        master_config["plan_grpo"]["num_judgments_per_plan"],
                    )
                    logger.log_table("plan_grpo_samples", log_data, step)

            # Update counters
            step += 1
            consumed_samples += batch.size
            plan_grpo_save_state["step"] = step
            plan_grpo_save_state["optim_step"] = optim_step
            plan_grpo_save_state["consumed_samples"] = consumed_samples

            # ============== Validation ==============
            if val_dataloader and val_period > 0 and step % val_period == 0:
                print("\n🔍 Running validation...")
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(policy, policy_generation, refit_buffer_size_gb)
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()
                val_metrics, validation_timings = validate(
                    policy_generation,
                    val_dataloader,
                    tokenizer,
                    val_task_to_env,
                    step=step,
                    master_config=master_config,
                    logger=logger,
                )
                policy_generation.finish_generation()
                logger.log_metrics(val_metrics, step, prefix="validation")
                logger.log_metrics(validation_timings, step, prefix="timing/validation")

            # ============== Checkpointing ==============
            if checkpointer.should_checkpoint(step):
                print("\n💾 Saving checkpoint...")
                with timer.time("checkpointing"):
                    checkpointer.save_checkpoint(
                        step=step,
                        policy=policy,
                        plan_grpo_state=plan_grpo_save_state,
                    )
                print(f"  ✓ Checkpoint saved at step {step}")

    print("\n" + "=" * 60)
    print(" " * 18 + "TRAINING COMPLETE")
    print("=" * 60 + "\n")


def combine_training_data(
    stage1_train_data: BatchedDataDict[ClippedPGLossDataDict],
    stage2_train_data: BatchedDataDict[ClippedPGLossDataDict],
) -> BatchedDataDict[ClippedPGLossDataDict]:
    """Combine training data from both stages.
    
    Note: Stage 2 (judgment) sequences might be longer than stage 1, so we pad appropriately.
    """
    
    # Get sequence lengths
    stage1_seq_len = stage1_train_data["input_ids"].shape[1]
    stage2_seq_len = stage2_train_data["input_ids"].shape[1]
    
    # Pad the shorter sequence to match the longer one
    if stage1_seq_len < stage2_seq_len:
        pad_size = stage2_seq_len - stage1_seq_len
        stage1_train_data["input_ids"] = torch.nn.functional.pad(
            stage1_train_data["input_ids"], (0, pad_size), value=0
        )
        stage1_train_data["advantages"] = torch.nn.functional.pad(
            stage1_train_data["advantages"], (0, pad_size), value=0
        )
        stage1_train_data["generation_logprobs"] = torch.nn.functional.pad(
            stage1_train_data["generation_logprobs"], (0, pad_size), value=0
        )
        stage1_train_data["token_mask"] = torch.nn.functional.pad(
            stage1_train_data["token_mask"], (0, pad_size), value=0
        )
    elif stage2_seq_len < stage1_seq_len:
        pad_size = stage1_seq_len - stage2_seq_len
        stage2_train_data["input_ids"] = torch.nn.functional.pad(
            stage2_train_data["input_ids"], (0, pad_size), value=0
        )
        stage2_train_data["advantages"] = torch.nn.functional.pad(
            stage2_train_data["advantages"], (0, pad_size), value=0
        )
        stage2_train_data["generation_logprobs"] = torch.nn.functional.pad(
            stage2_train_data["generation_logprobs"], (0, pad_size), value=0
        )
        stage2_train_data["token_mask"] = torch.nn.functional.pad(
            stage2_train_data["token_mask"], (0, pad_size), value=0
        )
    
    # Now concatenate the tensors (both have same sequence length)
    combined_data = BatchedDataDict[ClippedPGLossDataDict]({
        "input_ids": torch.cat([stage1_train_data["input_ids"], stage2_train_data["input_ids"]], dim=0),
        "input_lengths": torch.cat([stage1_train_data["input_lengths"], stage2_train_data["input_lengths"]], dim=0),
        "advantages": torch.cat([stage1_train_data["advantages"], stage2_train_data["advantages"]], dim=0),
        "generation_logprobs": torch.cat([stage1_train_data["generation_logprobs"], stage2_train_data["generation_logprobs"]], dim=0),
        "token_mask": torch.cat([stage1_train_data["token_mask"], stage2_train_data["token_mask"]], dim=0),
        "sample_mask": torch.cat([stage1_train_data["sample_mask"], stage2_train_data["sample_mask"]], dim=0),
    })
    
    # Shuffle the combined data to mix stage 1 and stage 2 samples
    total_samples = combined_data["input_ids"].shape[0]
    shuffle_indices = torch.randperm(total_samples)
    
    # Apply shuffle to all tensors
    combined_data["input_ids"] = combined_data["input_ids"][shuffle_indices]
    combined_data["input_lengths"] = combined_data["input_lengths"][shuffle_indices]
    combined_data["advantages"] = combined_data["advantages"][shuffle_indices]
    combined_data["generation_logprobs"] = combined_data["generation_logprobs"][shuffle_indices]
    combined_data["token_mask"] = combined_data["token_mask"][shuffle_indices]
    combined_data["sample_mask"] = combined_data["sample_mask"][shuffle_indices]
    
    # Move to CPU as grpo does
    combined_data.to("cpu")
    
    return combined_data



