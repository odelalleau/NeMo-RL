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
def extract_parallel_thinking_log_data(
    batch: BatchedDataDict[DatumSpec],
    stage1_repeated_batch: BatchedDataDict[DatumSpec],
    stage1_rewards: torch.Tensor,
    stage2_repeated_batch: BatchedDataDict[DatumSpec],
    stage2_rewards: torch.Tensor,
    num_generations_per_prompt: int,
) -> Dict[str, Any]:
    """Extract comprehensive logging data for parallel thinking training.
    
    Returns a dictionary with stage1/stage2 prompts, responses, and rewards for each original prompt.
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
    
    # Extract stage 1 data (first response for each original prompt)
    stage1_prompts = []
    stage1_responses = []
    stage1_rewards_first = []
    
    for i in range(num_original_prompts):
        first_gen_idx = i * num_generations_per_prompt
        prompt, response = extract_prompt_response(stage1_repeated_batch["message_log"][first_gen_idx])
        stage1_prompts.append(prompt)
        stage1_responses.append(response)
        stage1_rewards_first.append(stage1_rewards[first_gen_idx].item())
    
    # Extract stage 2 data
    stage2_prompts = []
    stage2_responses = []
    stage2_rewards_first = []
    num_stage2_prompts = len(stage2_repeated_batch["message_log"]) // num_generations_per_prompt
    
    for i in range(num_original_prompts):
        if i < num_stage2_prompts:
            first_gen_idx = i * num_generations_per_prompt
            prompt, response = extract_prompt_response(stage2_repeated_batch["message_log"][first_gen_idx])
            stage2_prompts.append(prompt)
            stage2_responses.append(response)
            stage2_rewards_first.append(stage2_rewards[first_gen_idx].item())
        else:
            stage2_prompts.append("N/A")
            stage2_responses.append("N/A")
            stage2_rewards_first.append(0.0)
    
    # Create log data with parallel thinking fields
    log_data = {
        "stage1_prompt": stage1_prompts,
        "stage1_response": stage1_responses,
        "stage1_reward": stage1_rewards_first,
        "stage2_prompt": stage2_prompts,
        "stage2_response": stage2_responses,
        "stage2_reward": stage2_rewards_first,
    }
    
    return log_data


def calculate_best_at_k_advantages_bootstrap(
    rewards: torch.Tensor,
    num_prompts: int,
    num_generations_per_prompt: int,
    k: int,
    m: int,
    normalize: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Calculate Best@k advantages using bootstrap sampling.
    
    Args:
        rewards: Tensor of shape (num_prompts * num_generations_per_prompt,) containing rewards
        num_prompts: Number of original prompts
        num_generations_per_prompt: Number of generations per prompt (n)
        k: Number of samples to select in each group
        m: Number of bootstrap groups to create
        normalize: Whether to normalize advantages
        
    Returns:
        Tuple of (advantages, metrics) where advantages has same shape as rewards
    """
    # Validate k
    if k > num_generations_per_prompt:
        raise ValueError(f"k ({k}) cannot be greater than num_generations_per_prompt ({num_generations_per_prompt})")
    
    # Calculate maximum possible unique groups
    max_unique_groups = math.comb(num_generations_per_prompt, k)
    
    # Decide sampling strategy
    use_all_combinations = max_unique_groups <= m
    effective_m = max_unique_groups if use_all_combinations else m
    
    if use_all_combinations:
        print(f"    Using all C({num_generations_per_prompt},{k})={max_unique_groups} unique combinations (< m={m})")
    else:
        print(f"    Bootstrap sampling m={m} groups from C({num_generations_per_prompt},{k})={max_unique_groups} possible combinations")
    
    # Reshape rewards to (num_prompts, num_generations_per_prompt)
    rewards_by_prompt = rewards.view(num_prompts, num_generations_per_prompt)
    
    # Initialize advantages with zeros
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    
    # Metrics to track
    group_rewards_all = []
    
    # Process each prompt separately
    for prompt_idx in range(num_prompts):
        prompt_rewards = rewards_by_prompt[prompt_idx]
        prompt_advantages = torch.zeros(num_generations_per_prompt, dtype=torch.float32)
        
        # Track which groups each sample belongs to
        sample_group_memberships = [[] for _ in range(num_generations_per_prompt)]
        group_rewards = []
        
        if use_all_combinations:
            # Generate all unique combinations
            all_indices = list(range(num_generations_per_prompt))
            all_combinations = list(combinations(all_indices, k))
            
            for group_idx, sampled_indices in enumerate(all_combinations):
                sampled_indices_tensor = torch.tensor(sampled_indices, dtype=torch.long)
                
                # Get the best reward in this group
                group_reward = prompt_rewards[sampled_indices_tensor].max().item()
                group_rewards.append(group_reward)
                
                # Track which samples are in this group
                for idx in sampled_indices:
                    sample_group_memberships[idx].append(group_idx)
        else:
            # Original bootstrap sampling with duplicate detection
            seen_combinations = set()
            actual_groups_created = 0
            attempts = 0
            max_attempts = m * 3  # Prevent infinite loop
            
            while actual_groups_created < m and attempts < max_attempts:
                attempts += 1
                
                # Randomly sample k indices without replacement
                sampled_indices = torch.randperm(num_generations_per_prompt)[:k]
                
                # Check if we've seen this combination before
                indices_tuple = tuple(sorted(sampled_indices.tolist()))
                if indices_tuple in seen_combinations:
                    continue
                seen_combinations.add(indices_tuple)
                
                # Get the best reward in this group
                group_reward = prompt_rewards[sampled_indices].max().item()
                group_rewards.append(group_reward)
                
                # Track which samples are in this group
                for idx in sampled_indices:
                    sample_group_memberships[idx.item()].append(actual_groups_created)
                
                actual_groups_created += 1
        
        # Calculate baseline and advantages for groups
        group_rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)
        group_baseline = group_rewards_tensor.mean()
        group_advantages = group_rewards_tensor - group_baseline
        
        # Normalize group advantages if requested
        if normalize:
            group_std = group_rewards_tensor.std()
            if group_std > 0:
                group_advantages = group_advantages / group_std
        
        # Assign advantages to samples based on their group memberships
        for sample_idx in range(num_generations_per_prompt):
            if sample_group_memberships[sample_idx]:
                # Sum advantages from all groups this sample belongs to
                sample_advantage = sum(
                    group_advantages[group_idx].item() 
                    for group_idx in sample_group_memberships[sample_idx]
                )
                prompt_advantages[sample_idx] = sample_advantage
        
        # Store advantages for this prompt
        start_idx = prompt_idx * num_generations_per_prompt
        end_idx = start_idx + num_generations_per_prompt
        advantages[start_idx:end_idx] = prompt_advantages
        
        group_rewards_all.extend(group_rewards)
    
    # Calculate metrics
    metrics = {
        "best_at_k_mean_group_reward": np.mean(group_rewards_all),
        "best_at_k_std_group_reward": np.std(group_rewards_all),
        "best_at_k_min_group_reward": np.min(group_rewards_all),
        "best_at_k_max_group_reward": np.max(group_rewards_all),
        "best_at_k_k": k,
        "best_at_k_m": m,
        "best_at_k_effective_m": effective_m,
        "best_at_k_max_unique_groups": max_unique_groups,
        "best_at_k_used_all_combinations": use_all_combinations,
    }
    
    return advantages.unsqueeze(-1), metrics


def detect_stage1_majority_clusters(
    stage1_rewards: torch.Tensor,
    num_prompts: int,
    num_generations_per_prompt: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detect majority clusters in Stage 1 responses based on reward values.
    
    A majority cluster exists when one reward value has strictly more responses
    than any other reward value (no ties).
    
    Args:
        stage1_rewards: Tensor of shape (num_prompts * num_generations_per_prompt,)
        num_prompts: Number of original prompts
        num_generations_per_prompt: Number of generations per prompt
        
    Returns:
        Tuple of:
        - has_majority: Boolean tensor of shape (num_prompts,) indicating which prompts have majority
        - majority_reward: Tensor of shape (num_prompts,) with majority reward value (0 if no majority)
        - majority_size: Tensor of shape (num_prompts,) with size of majority cluster (0 if no majority)
    """
    # Reshape to (num_prompts, num_generations_per_prompt)
    rewards_by_prompt = stage1_rewards.view(num_prompts, num_generations_per_prompt)
    
    has_majority = torch.zeros(num_prompts, dtype=torch.bool)
    majority_reward = torch.zeros(num_prompts, dtype=torch.float32)
    majority_size = torch.zeros(num_prompts, dtype=torch.long)
    
    for prompt_idx in range(num_prompts):
        prompt_rewards = rewards_by_prompt[prompt_idx].tolist()
        
        # Count occurrences of each reward value
        reward_counts = Counter(prompt_rewards)
        
        # Sort by count (descending)
        sorted_counts = sorted(reward_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Check if there's a strict majority (no ties at the top)
        if len(sorted_counts) >= 2 and sorted_counts[0][1] > sorted_counts[1][1]:
            has_majority[prompt_idx] = True
            majority_reward[prompt_idx] = sorted_counts[0][0]
            majority_size[prompt_idx] = sorted_counts[0][1]
        elif len(sorted_counts) == 1:
            # All responses have the same reward - this is a majority
            has_majority[prompt_idx] = True
            majority_reward[prompt_idx] = sorted_counts[0][0]
            majority_size[prompt_idx] = sorted_counts[0][1]
    
    return has_majority, majority_reward, majority_size


def apply_environment_post_processing(
    batch: BatchedDataDict[DatumSpec], 
    task_to_env: Dict[str, EnvironmentInterface],
    prefix: str = ""
) -> Tuple[BatchedDataDict[DatumSpec], Dict[str, Any]]:
    """Apply environment post-processing to a batch.
    
    Args:
        batch: Batch containing completed rollout data
        task_to_env: Dictionary mapping task names to their corresponding environments
        prefix: Optional prefix for metric names (e.g., "val_" for validation metrics)
        
    Returns:
        Tuple of (original_batch, environment_metrics)
        
    Note:
        This function only collects metrics from environment post-processing.
        The batch is returned unchanged since training data was already processed during rollouts.
    """
    import ray
    
    env_metrics = {}
    
    # Collect futures for all environment post-processing calls
    futures = []
    task_info = []  # Store task_name for each future
    
    for task_name, env in task_to_env.items():
        # Find indices for this task type
        task_indices = [i for i, name in enumerate(batch["task_name"]) if name == task_name]
        if task_indices:
            # Create a sub-batch for this task
            task_batch = batch.select_indices(task_indices)
            
            # Apply environment post-processing (Ray remote call)
            future = env.global_post_process_and_metrics.remote(task_batch)
            futures.append(future)
            task_info.append(task_name)
    
    # Get results and collect only metrics (ignore processed batches)
    if futures:
        results = ray.get(futures)
        
        # Process results - only collect metrics, ignore batch modifications
        for task_name, (_, task_env_metrics) in zip(task_info, results):
            # Collect metrics with appropriate prefix
            metric_prefix = f"{prefix}{task_name}/env_" if prefix else f"{task_name}/env_"
            env_metrics.update({f"{metric_prefix}{k}": v for k, v in task_env_metrics.items()})
    
    # Return original batch unchanged, just with additional metrics
    return batch, env_metrics


# ===============================================================================
# Configuration
# ===============================================================================
@dataclass
class TimeLimitTimer:
    """Timer to tell us when the time limit is reached"""

    duration: Optional[str]

    def __post_init__(self):
        self._duration = float("inf")

        if self.duration is not None:
            days, hours, mins, seconds = map(int, self.duration.strip().split(":"))
            self._duration = timedelta(
                days=days, hours=hours, minutes=mins, seconds=seconds
            ).total_seconds()

    def start_time(self):
        self._start_time = time.monotonic()

    def get_time_elapsed(self):
        return time.monotonic() - self._start_time

    def get_time_remaining(self):
        return self._duration - self.get_time_elapsed()

    def is_finished(self):
        time_left = self.get_time_remaining()
        return time_left <= 0


class ParallelThinkingGRPOConfig(TypedDict):
    num_prompts_per_step: int
    num_generations_per_prompt: int
    normalize_rewards: bool
    use_leave_one_out_baseline: bool
    val_period: int
    val_batch_size: int
    val_at_start: bool
    checkpoint_dir: str
    num_epochs: int
    max_rollout_turns: int
    max_val_samples: int
    num_val_repeats: int  # Number of validation generations per prompt
    time_limit: Optional[str]  # Time limit in format "days:hours:mins:seconds"
    check_baseline_correctness: bool  # Enable baseline correctness assertions
    # Parallel thinking specific configuration
    aggregation_prompt_template: str  # Template for aggregation prompts
    skip_stage1_env_post_processing: bool # Skip environment post-processing for stage 1
    skip_stage2_env_post_processing: bool # Skip environment post-processing for stage 2
    skip_val_env_post_processing: bool # Skip environment post-processing for validation
    # Best@k training configuration for stage 1
    use_best_at_k_for_stage1: bool  # Enable Best@k training for stage 1
    best_at_k_k: int  # k value for Best@k (number of samples per group)
    best_at_k_m: int  # m value for Best@k (number of bootstrap groups)
    # Binary reward configuration for stage 2
    use_binary_reward_for_stage2: bool  # Enable binary reward transformation for stage 2
    binary_reward_threshold_type: str  # Type of threshold: "best" (vs best stage1) or future options
    # Cluster handling - proportional bonus for overcoming Stage 1 clusters
    use_stage2_cluster_bonus: bool  # Enable proportional bonus for overcoming Stage 1 clusters (default: False)
    stage2_cluster_bonus_scale: float  # Scale factor for proportional bonus based on cluster size (default: 1.0)


class ParallelThinkingGRPOSaveState(TypedDict):
    step: int
    optim_step: int
    val_reward: float
    consumed_samples: int


def _default_pt_grpo_save_state() -> ParallelThinkingGRPOSaveState:
    return {
        "step": 0,
        "optim_step": 0,
        "val_reward": -99999999.0,
        "consumed_samples": 0,
    }


class MasterConfig(TypedDict):
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig
    env: Dict[str, Any]
    data: DataConfig
    pt_grpo: ParallelThinkingGRPOConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# ===============================================================================
# Setup & Initialization
# ===============================================================================


def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> Tuple[
    PolicyInterface,
    GenerationInterface,
    RayVirtualCluster,
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    ClippedPGLossFn,
    Logger,
    CheckpointManager,
    ParallelThinkingGRPOSaveState,
    MasterConfig,
]:
    """Main entry point for running Parallel Thinking GRPO algorithm.

    Returns:
        Tuple of policy, cluster, dataloader, tokenizer, loss_fn, logger, checkpointer, save_state, master_config, val_dataloader
    """
    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    data_config = master_config["data"]
    pt_grpo_config = master_config["pt_grpo"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    pt_grpo_save_state: Optional[ParallelThinkingGRPOSaveState] = checkpointer.load_training_info(
        last_checkpoint_path
    )
    if pt_grpo_save_state is None:
        pt_grpo_save_state = _default_pt_grpo_save_state()

    # config validation checks
    if master_config["checkpointing"]["enabled"]:
        assert master_config["checkpointing"]["save_period"] > 0
        assert (
            master_config["checkpointing"]["save_period"]
            % master_config["pt_grpo"]["val_period"]
            == 0
        ), (
            f"Checkpointing save period {master_config['checkpointing']['save_period']} "
            f"must be a multiple of validation period {master_config['pt_grpo']['val_period']}"
            f", or we won't know what metric to save!"
        )

    # ==========================
    #           Data
    # ==========================
    shuffle_train = master_config["data"]["train"]["shuffle"]
    shuffle_val = master_config["data"]["val"]["shuffle"]

    train_data_generator = None
    val_data_generator = None

    if shuffle_train:
        train_data_generator = torch.Generator()
        train_data_generator.manual_seed(master_config["data"]["train"]["seed"])

    if shuffle_val:
        val_data_generator = torch.Generator()
        val_data_generator.manual_seed(master_config["data"]["val"]["seed"])

    dataloader = StatefulDataLoader(
        dataset,
        batch_size=pt_grpo_config["num_prompts_per_step"],
        shuffle=shuffle_train,
        generator=train_data_generator,
        collate_fn=rl_collate_fn,
        drop_last=master_config["data"]["train"]["drop_last"],
    )
    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        dataloader.load_state_dict(dataloader_state_dict)

    print(f"  ✓ Training dataloader loaded with {len(dataset)} samples")

    # Load validation dataset if provided
    val_dataloader = None
    # If validation is enabled, load the validation dataloader
    if pt_grpo_config["val_period"] > 0 or pt_grpo_config["val_at_start"]:
        val_batch_size = min(master_config["pt_grpo"]["max_val_samples"], len(val_dataset))
        if "val_batch_size" in master_config["pt_grpo"]:
            print("val batch size is specified but we don't actually use it anymore")

        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=shuffle_val,
            collate_fn=rl_collate_fn,
            generator=val_data_generator,
            drop_last=master_config["data"]["val"]["drop_last"],
        )
        print(f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples")

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    colocated_inference = generation_config["backend"] != "hf"
    cluster = RayVirtualCluster(
        name="pt_grpo_policy_cluster",
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

    policy = HfPolicy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
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
        pt_grpo_save_state,
        master_config,
    )


def get_reasoning_split_word(env_configs: Dict[str, Any]) -> Optional[str]:
    """Get reasoning_split_word from any enabled environment."""
    for env_name, env_config in env_configs.items():
        if env_config.get("enable", False) and "reasoning_split_word" in env_config:
            return env_config["reasoning_split_word"]
    return None


def create_aggregation_prompts(
    original_batch: BatchedDataDict[DatumSpec],
    stage1_responses: List[List[str]],
    aggregation_prompt_template: str,
    tokenizer,
) -> BatchedDataDict[DatumSpec]:
    """Create aggregation prompts by combining original prompts with stage 1 responses.
    
    Args:
        original_batch: Original batch containing the prompts
        stage1_responses: List of lists, where each inner list contains the stage 1 responses for a prompt
        aggregation_prompt_template: Template string for formatting aggregation prompts
        tokenizer: Tokenizer to use for tokenizing the aggregation prompts
    
    Returns:
        New batch with aggregation prompts
    """
    aggregation_batch = deepcopy(original_batch)
    
    # Create new message logs for aggregation
    new_message_logs = []
    new_extra_env_info = []
    new_loss_multiplier = []
    new_task_name = []
    
    for i, message_log in enumerate(original_batch["message_log"]):
        # Skip if no responses for this prompt
        if len(stage1_responses[i]) == 0:
            continue
            
        # Extract the original user question from metadata
        if "extra_env_info" not in original_batch or not original_batch["extra_env_info"][i]:
            raise ValueError(f"No extra_env_info found in batch for sample {i}")
        
        original_prompt = original_batch["extra_env_info"][i].get("question")
        if original_prompt is None:
            raise ValueError(f"No question found in metadata for sample {i}")
        
        # Format the stage 1 responses
        responses_text = ""
        for j, response in enumerate(stage1_responses[i]):
            responses_text += f"<Solution {j+1}>\n{response}\n</Solution {j+1}>\n"
        
        # Create the aggregation prompt using the template
        aggregation_prompt = aggregation_prompt_template.format(
            original_prompt=original_prompt,
            responses=responses_text.strip()
        )
        
        # Create a proper message structure for chat template
        aggregation_message = [
            {
                "role": "user",
                "content": aggregation_prompt,
            }
        ]
        
        # Apply chat template to get properly formatted content and token_ids
        formatted_content = tokenizer.apply_chat_template(
            aggregation_message,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        token_ids = tokenizer.apply_chat_template(
            aggregation_message,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=False,
            return_tensors="pt",
        )[0]
        
        # Create new message log with the aggregation prompt
        new_message_log = [
            {
                "role": "user",
                "content": formatted_content,
                "token_ids": token_ids,
            }
        ]
        new_message_logs.append(new_message_log)
    
        # Keep track of corresponding metadata
        new_extra_env_info.append(original_batch["extra_env_info"][i])
        new_loss_multiplier.append(original_batch["loss_multiplier"][i])
        new_task_name.append(original_batch["task_name"][i])
    
    # Ensure we have at least some valid aggregation prompts
    if len(new_message_logs) == 0:
        raise ValueError("No valid aggregation prompts could be created from the stage 1 responses")
    
    print(f"  ✓ Created {len(new_message_logs)} aggregation prompts from {len(original_batch['message_log'])} original prompts")
    
    # Update aggregation batch with new data
    aggregation_batch["message_log"] = new_message_logs
    aggregation_batch["extra_env_info"] = new_extra_env_info
    aggregation_batch["loss_multiplier"] = torch.tensor(new_loss_multiplier)
    aggregation_batch["task_name"] = new_task_name
    
    return aggregation_batch


def combine_and_shuffle_training_data(
    stage1_train_data: BatchedDataDict[ClippedPGLossDataDict],
    stage2_train_data: BatchedDataDict[ClippedPGLossDataDict],
) -> BatchedDataDict[ClippedPGLossDataDict]:
    """Combine training data from both stages and shuffle.
    
    Note: Stage 2 (aggregation) sequences are typically longer than stage 1, so we pad stage 1 data to match.
    The combined data is shuffled to mix stage 1 and stage 2 samples.
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
    
    return combined_data


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


def parallel_thinking_grpo_train(
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
    pt_grpo_save_state: Optional[ParallelThinkingGRPOSaveState],
    master_config: MasterConfig,
):
    """Run Parallel Thinking GRPO training algorithm."""
    timer = Timer()
    NEED_REFIT = True
    # If policy_generation is None, use the policy as the generation interface (hf framework backend)
    if policy_generation is None:
        policy_generation = policy
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running

    # common config/state items
    step = pt_grpo_save_state["step"]
    optim_step = pt_grpo_save_state["optim_step"]

    consumed_samples = pt_grpo_save_state["consumed_samples"]
    val_period = master_config["pt_grpo"]["val_period"]
    val_at_start = master_config["pt_grpo"]["val_at_start"]
    refit_buffer_size_gb = master_config["policy"]["refit_buffer_size_gb"]

    num_epochs = master_config["pt_grpo"]["num_epochs"]
    max_num_steps = num_epochs * len(dataloader)

    # Initialize time limit timer
    time_limit_timer = TimeLimitTimer(master_config["pt_grpo"].get("time_limit"))
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

    # Run parallel thinking GRPO training
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
            # ============== Stage 1: Normal Generation ==============
            print("\n▶ Stage 1: Normal Generation...")
            with timer.time("stage1_preparation"):
                # Repeat batch items for multiple generations
                stage1_repeated_batch: BatchedDataDict[DatumSpec] = batch.repeat_interleave(
                    master_config["pt_grpo"]["num_generations_per_prompt"]
                )
                # Convert LLMMessageLogType to FlatMessagesType for generation and save prompt ids
                batched_flat_pre_gen, input_lengths = batched_message_log_to_flat_message(
                    stage1_repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                )
                stage1_prompt_input_ids = batched_flat_pre_gen["token_ids"]  # prompt-only ids (no assistant)

            # Generate responses for stage 1
            print(f"  • Generating {stage1_repeated_batch.size} stage 1 responses...")
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
                    max_rollout_turns=master_config["pt_grpo"]["max_rollout_turns"],
                    greedy=False,
                )
                
                # Keep generation active for stage 2

            # Apply environment post-processing for stage 1
            print("  • Applying stage 1 environment post-processing...")
            with timer.time("stage1_env_post_processing"):
                if master_config["pt_grpo"].get("skip_stage1_env_post_processing", False):
                    stage1_env_metrics = {}
                    print("    (Skipped - skip_stage1_env_post_processing is enabled)")
                else:
                    stage1_repeated_batch, stage1_env_metrics = apply_environment_post_processing(
                        stage1_repeated_batch, task_to_env, prefix="stage1_"
                    )
                stage1_rollout_metrics.update(stage1_env_metrics)

            # Get dataset specific pass at k for stage 1
            stage1_prompt_based_reward_dict = defaultdict(list)
            stage1_idx_dictionary = defaultdict(list)
            if "dataset_names" in stage1_repeated_batch and "idx" in stage1_repeated_batch:
                for dataset, r, idx in zip(
                    stage1_repeated_batch["dataset_names"],
                    stage1_repeated_batch["total_reward"],
                    stage1_repeated_batch["idx"],
                ):
                    stage1_prompt_based_reward_dict[dataset].append(r)
                    stage1_idx_dictionary[dataset].append(idx)

                for dataset, rewards in stage1_prompt_based_reward_dict.items():
                    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32).view(
                        -1, master_config["pt_grpo"]["num_generations_per_prompt"]
                    )
                    stage1_rollout_metrics[
                        f"stage1_{dataset}/pass_at_{master_config['pt_grpo']['num_generations_per_prompt']}"
                    ] = (rewards_tensor > 0).any(-1).float().mean()

            # Extract stage 1 responses and rewards
            stage1_rewards = stage1_repeated_batch["total_reward"]
            
            # ============== Stage 2: Aggregation ==============
            print("\n▶ Stage 2: Aggregation...")
            with timer.time("stage2_preparation"):
                # Extract stage 1 responses grouped by original prompt
                num_prompts = len(batch["message_log"])
                num_generations = master_config["pt_grpo"]["num_generations_per_prompt"]
                
                # Get reasoning split word from any enabled environment
                reasoning_split_word = get_reasoning_split_word(master_config["env"])
                
                # Group responses by original prompt
                stage1_responses = []
                for i in range(num_prompts):
                    prompt_responses = []
                    for j in range(num_generations):
                        idx = i * num_generations + j
                        # Extract assistant response from the message log
                        last_assistant_response = None
                        for message in stage1_repeated_batch["message_log"][idx]:
                            if message["role"] == "assistant":
                                last_assistant_response = message["content"]
                        if last_assistant_response is not None:
                            if reasoning_split_word and reasoning_split_word in last_assistant_response:
                                # Remove reasoning part if split word exists
                                prompt_responses.append(last_assistant_response.split(reasoning_split_word)[-1].lstrip()[:6000])
                            else:
                                prompt_responses.append(last_assistant_response[:6000])
                    
                    # Randomly select a subset of responses for aggregation
                    # Only select powers of 2: 1, 2, 4, 8, etc.
                    num_to_select = random.choice([2**i for i in range(len(prompt_responses).bit_length()) if 2**i <= len(prompt_responses)])
                    selected_responses = random.sample(prompt_responses, num_to_select)
                    stage1_responses.append(selected_responses)
                
                # Create aggregation prompts
                aggregation_batch_template = create_aggregation_prompts(
                    batch,
                    stage1_responses, 
                    master_config["pt_grpo"]["aggregation_prompt_template"],
                    tokenizer,
                )
                
                # Repeat aggregation batch for multiple generations
                stage2_repeated_batch = aggregation_batch_template.repeat_interleave(
                    master_config["pt_grpo"]["num_generations_per_prompt"]
                )
                
                # Calculate input_ids for stage 2
                stage2_flat_pre_rollout, stage2_input_lengths = batched_message_log_to_flat_message(
                    stage2_repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                )
                stage2_prompt_input_ids = stage2_flat_pre_rollout["token_ids"]

            print(f"  • Generating {stage2_repeated_batch.size} stage 2 aggregation responses...")
            with timer.time("stage2_generation"):
                stage2_repeated_batch, stage2_rollout_metrics = run_multi_turn_rollout(
                    policy_generation=policy_generation,
                    input_batch=stage2_repeated_batch,
                    tokenizer=tokenizer,
                    task_to_env=task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["pt_grpo"]["max_rollout_turns"],
                    greedy=False,
                )
                
                policy_generation.finish_generation()

            # Apply environment post-processing for stage 2
            print("  • Applying stage 2 environment post-processing...")
            with timer.time("stage2_env_post_processing"):
                if master_config["pt_grpo"].get("skip_stage2_env_post_processing", False):
                    stage2_env_metrics = {}
                    print("    (Skipped - skip_stage2_env_post_processing is enabled)")
                else:
                    stage2_repeated_batch, stage2_env_metrics = apply_environment_post_processing(
                        stage2_repeated_batch, task_to_env, prefix="stage2_"
                    )
                stage2_rollout_metrics.update(stage2_env_metrics)

            # Get dataset specific pass at k for stage 2
            stage2_prompt_based_reward_dict = defaultdict(list)
            stage2_idx_dictionary = defaultdict(list)
            if "dataset_names" in stage2_repeated_batch and "idx" in stage2_repeated_batch:
                for dataset, r, idx in zip(
                    stage2_repeated_batch["dataset_names"],
                    stage2_repeated_batch["total_reward"],
                    stage2_repeated_batch["idx"],
                ):
                    stage2_prompt_based_reward_dict[dataset].append(r)
                    stage2_idx_dictionary[dataset].append(idx)

                for dataset, rewards in stage2_prompt_based_reward_dict.items():
                    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32).view(
                        -1, master_config["pt_grpo"]["num_generations_per_prompt"]
                    )
                    stage2_rollout_metrics[
                        f"stage2_{dataset}/pass_at_{master_config['pt_grpo']['num_generations_per_prompt']}"
                    ] = (rewards_tensor > 0).any(-1).float().mean()

            # Extract stage 2 rewards
            stage2_rewards = stage2_repeated_batch["total_reward"]

            # Transform stage 2 rewards to binary if configured
            stage2_rewards_original = stage2_rewards.clone()  # Keep original for metrics
            binary_reward_rate = None
            
            if master_config["pt_grpo"].get("use_binary_reward_for_stage2", False):
                print("  • Applying binary reward transformation for stage 2...")
                
                # Reshape rewards by prompt
                stage1_rewards_by_prompt = stage1_rewards.view(num_prompts, num_generations)
                stage2_rewards_by_prompt = stage2_rewards.view(num_prompts, num_generations)
                
                threshold_type = master_config["pt_grpo"].get("binary_reward_threshold_type", "best")
                
                if threshold_type == "best":
                    # Get best stage 1 reward for each prompt
                    best_stage1_per_prompt = stage1_rewards_by_prompt.max(dim=1, keepdim=True)[0]
                    
                    # Create binary rewards: 1 if stage2 >= best stage1, 0 otherwise
                    stage2_binary_rewards = (stage2_rewards_by_prompt >= best_stage1_per_prompt).float()
                    stage2_rewards = stage2_binary_rewards.view(-1)  # Flatten back
                    
                    # Log binary reward stats
                    binary_reward_rate = stage2_binary_rewards.mean().item()
                    print(f"    Binary reward rate: {binary_reward_rate:.2%} of stage 2 responses beat/match best stage 1")
                else:
                    raise ValueError(f"Unknown binary_reward_threshold_type: {threshold_type}")

            # Calculate stage2_better_than_stage1_avg_rate metric (vectorized, strict shape check)
            # Enforce that Stage 2 has the same number of prompts as Stage 1; otherwise raise an error.
            stage2_num_prompts = stage2_rewards.numel() // num_generations if stage2_rewards.numel() > 0 else 0
            if stage2_num_prompts != num_prompts:
                raise ValueError(
                    f"Stage 2 prompt count ({stage2_num_prompts}) does not match Stage 1 prompt count ({num_prompts}). "
                    "Likely some prompts produced zero valid Stage 1 responses and were skipped during aggregation."
                )

            # Group rewards by prompt
            stage1_rewards_by_prompt = stage1_rewards.view(num_prompts, num_generations)
            stage2_rewards_by_prompt = stage2_rewards_original.view(num_prompts, num_generations)  # Use original rewards

            # Average stage1 rewards per prompt
            stage1_avg_rewards_per_prompt = stage1_rewards_by_prompt.mean(dim=1, keepdim=True)

            # Compare each stage2 response to its prompt's stage1 average and take overall mean
            stage2_better_than_stage1_avg_rate = (
                (stage2_rewards_by_prompt >= stage1_avg_rewards_per_prompt).float().mean().item()
            )

            # ============== Calculate Majority Cluster Metrics ==============
            print("\n▶ Detecting Stage 1 majority clusters...")
            has_majority, majority_reward, majority_size = detect_stage1_majority_clusters(
                stage1_rewards, num_prompts, num_generations
            )
            
            # Calculate metrics
            majority_rate = has_majority.float().mean().item()
            print(f"  • {majority_rate:.2%} of prompts have a majority cluster in Stage 1")
            
            # For prompts with majority, check Stage 2 performance
            if has_majority.any():
                # Get Stage 2 rewards for prompts with majority (take first generation per prompt as representative)
                stage2_first_gen_rewards = stage2_rewards_original.view(num_prompts, num_generations)[:, 0]
                
                # Stage 2 same as majority
                stage2_same_as_majority = torch.zeros(num_prompts, dtype=torch.bool)
                stage2_same_as_majority[has_majority] = (
                    stage2_first_gen_rewards[has_majority] == majority_reward[has_majority]
                )
                stage2_same_as_majority_rate = stage2_same_as_majority[has_majority].float().mean().item()
                
                # Stage 2 better than majority
                stage2_better_than_majority = torch.zeros(num_prompts, dtype=torch.bool)
                stage2_better_than_majority[has_majority] = (
                    stage2_first_gen_rewards[has_majority] > majority_reward[has_majority]
                )
                stage2_better_than_majority_rate = stage2_better_than_majority[has_majority].float().mean().item()
                
                print(f"  • Among prompts with majority:")
                print(f"    - {stage2_same_as_majority_rate:.2%} of Stage 2 responses match majority reward")
                print(f"    - {stage2_better_than_majority_rate:.2%} of Stage 2 responses beat majority reward")
                
                # Average majority cluster size for prompts with majority
                avg_majority_size = majority_size[has_majority].float().mean().item()
                print(f"    - Average majority cluster size: {avg_majority_size:.1f}/{num_generations}")
            else:
                stage2_same_as_majority_rate = 0.0
                stage2_better_than_majority_rate = 0.0
                avg_majority_size = 0.0
            
            # ============== Add Bonus to Stage 2 Rewards for Overcoming Majority Clusters ==============
            use_stage2_cluster_bonus = master_config["pt_grpo"].get("use_stage2_cluster_bonus", False)
            stage2_cluster_bonus_scale = master_config["pt_grpo"].get("stage2_cluster_bonus_scale", 1.0)
            
            if use_stage2_cluster_bonus and has_majority.any() and stage2_cluster_bonus_scale > 0:
                print(f"\n▶ Adding proportional reward bonus for Stage 2 responses that overcome Stage 1 clusters...")
                
                # For each stage 2 response, check if it beats the corresponding stage 1 majority
                stage2_rewards_by_prompt = stage2_rewards_original.view(num_prompts, num_generations)
                
                bonus_count = 0
                total_bonus_added = 0.0
                bonus_details = []
                
                for prompt_idx in range(num_prompts):
                    if has_majority[prompt_idx]:
                        # This prompt has a majority cluster in Stage 1
                        stage1_majority_reward = majority_reward[prompt_idx]
                        cluster_fraction = majority_size[prompt_idx].float() / num_generations
                        
                        # Calculate bonus proportional to cluster size
                        # For example, if 6/8 responses are the same, cluster_fraction = 0.75
                        # The bonus would be 0.75 * stage2_cluster_bonus_scale
                        proportional_bonus = cluster_fraction * stage2_cluster_bonus_scale
                        
                        # Check each Stage 2 generation for this prompt
                        for gen_idx in range(num_generations):
                            global_idx = prompt_idx * num_generations + gen_idx
                            stage2_reward = stage2_rewards_by_prompt[prompt_idx, gen_idx]
                            
                            # If this Stage 2 response beats the Stage 1 majority, add proportional bonus
                            if stage2_reward > stage1_majority_reward:
                                # Add bonus to the actual training reward (not the original)
                                stage2_rewards[global_idx] = stage2_rewards[global_idx] + proportional_bonus
                                bonus_count += 1
                                total_bonus_added += proportional_bonus
                                
                                # Track bonus details for logging
                                if len(bonus_details) < 5:  # Keep first 5 for example
                                    bonus_details.append(f"{majority_size[prompt_idx]}/{num_generations} -> +{proportional_bonus:.3f}")
                
                print(f"  • Added proportional bonuses to {bonus_count} Stage 2 responses")
                print(f"    Total bonus added: {total_bonus_added:.2f} (scale factor: {stage2_cluster_bonus_scale})")
                if bonus_details:
                    print(f"    Examples: {', '.join(bonus_details)}")
                
                # Update repeated batch with modified rewards
                stage2_repeated_batch["total_reward"] = stage2_rewards

            # ============== Calculate Rewards & Advantages ==============
            print("\n▶ Processing rewards and advantages...")
            with timer.time("reward_calculation"):
                # Stage 1 advantages
                print("  • Computing stage 1 advantages...")
                
                # Check for potentially problematic configuration
                expected_responses = master_config["pt_grpo"]["num_generations_per_prompt"] if master_config["pt_grpo"].get("check_baseline_correctness", True) else None
                
                # Use Best@k training for stage 1 if configured
                if master_config["pt_grpo"].get("use_best_at_k_for_stage1", False):
                    print("    Using Best@k bootstrap sampling for stage 1...")
                    k = master_config["pt_grpo"]["best_at_k_k"]
                    m = master_config["pt_grpo"]["best_at_k_m"]
                    
                    stage1_advantages, stage1_best_at_k_metrics = calculate_best_at_k_advantages_bootstrap(
                        rewards=stage1_rewards,
                        num_prompts=num_prompts,
                        num_generations_per_prompt=master_config["pt_grpo"]["num_generations_per_prompt"],
                        k=k,
                        m=m,
                        normalize=master_config["pt_grpo"]["normalize_rewards"],
                    )
                    
                    # Create stage1_baseline and stage1_std for compatibility
                    # not meaningful for best@k training
                    stage1_baseline = torch.zeros_like(stage1_rewards)
                    stage1_std = torch.ones_like(stage1_rewards)
                    stage1_metrics = stage1_best_at_k_metrics
                    
                    print(f"    Best@k: k={k}, m={m}, mean_group_reward={stage1_best_at_k_metrics['best_at_k_mean_group_reward']:.4f}")
                    print(f"    Max unique groups: C({master_config['pt_grpo']['num_generations_per_prompt']},{k})={stage1_best_at_k_metrics['best_at_k_max_unique_groups']}")
                    if stage1_best_at_k_metrics['best_at_k_used_all_combinations']:
                        print(f"    ✓ Used all {stage1_best_at_k_metrics['best_at_k_effective_m']} unique combinations")
                    else:
                        print(f"    • Sampled {stage1_best_at_k_metrics['best_at_k_effective_m']} groups")
                else:
                    # Original baseline calculation
                    stage1_baseline, stage1_std, stage1_metrics = calculate_baseline_and_std_per_prompt(
                        stage1_prompt_input_ids,
                        stage1_rewards,
                        torch.ones_like(stage1_rewards),
                        leave_one_out_baseline=master_config["pt_grpo"]["use_leave_one_out_baseline"],
                        expected_responses_per_prompt=expected_responses,
                    )
                    stage1_advantages = (stage1_rewards - stage1_baseline).unsqueeze(-1)
                    
                    # Normalize rewards if configured (only for non-Best@k)
                    if master_config["pt_grpo"]["normalize_rewards"]:
                        zero_std_mask = stage1_std > 0
                        stage1_advantages[zero_std_mask] = (
                            stage1_advantages[zero_std_mask] / stage1_std.unsqueeze(-1)[zero_std_mask]
                        )

                # Stage 2 advantages
                print("  • Computing stage 2 advantages...")
                stage2_baseline, stage2_std, stage2_metrics = calculate_baseline_and_std_per_prompt(
                    stage2_prompt_input_ids,
                    stage2_rewards,
                    torch.ones_like(stage2_rewards),
                    leave_one_out_baseline=master_config["pt_grpo"]["use_leave_one_out_baseline"],
                    expected_responses_per_prompt=expected_responses,
                )
                stage2_advantages = (stage2_rewards - stage2_baseline).unsqueeze(-1)

                # Normalize rewards if configured
                if master_config["pt_grpo"]["normalize_rewards"]:
                    # Stage 1 normalization - skip if using Best@k (already normalized)
                    if not master_config["pt_grpo"].get("use_best_at_k_for_stage1", False):
                        zero_std_mask = stage1_std > 0
                        stage1_advantages[zero_std_mask] = (
                            stage1_advantages[zero_std_mask] / stage1_std.unsqueeze(-1)[zero_std_mask]
                        )

                    # Stage 2 normalization
                    zero_std_mask = stage2_std > 0
                    stage2_advantages[zero_std_mask] = (
                        stage2_advantages[zero_std_mask] / stage2_std.unsqueeze(-1)[zero_std_mask]
                    )

                # Combine all rewards and advantages for metrics
                # Use original rewards for stage 2 when binary rewards are enabled
                if master_config["pt_grpo"].get("use_binary_reward_for_stage2", False):
                    all_rewards = torch.cat([stage1_rewards, stage2_rewards_original])  # Use original for display
                else:
                    all_rewards = torch.cat([stage1_rewards, stage2_rewards])
                all_advantages = torch.cat([stage1_advantages.flatten(), stage2_advantages.flatten()])
                
                # Calculate metrics
                rollout_metrics = {}
                rollout_metrics.update({"stage1_" + k: v for k, v in stage1_rollout_metrics.items()})
                rollout_metrics.update({"stage2_" + k: v for k, v in stage2_rollout_metrics.items()})
                rollout_metrics.update({"stage1_" + k: v for k, v in stage1_metrics.items()})
                rollout_metrics.update({"stage2_" + k: v for k, v in stage2_metrics.items()})
                
                # Stage-specific metrics
                rollout_metrics.update({
                    "stage1_reward_min": stage1_rewards.min(),
                    "stage1_reward_mean": stage1_rewards.mean(),
                    "stage1_reward_max": stage1_rewards.max(),
                    "stage1_baseline_mean": stage1_baseline.mean(),
                    "stage1_std_mean": stage1_std.mean(),
                    "stage1_perfect_prediction_rate": (stage1_rewards == 0).float().mean(),  # Percentage of perfect predictions
                    "stage1_percent_zero_advantages": (stage1_advantages == 0).float().mean(),  # Zero advantages for stage 1
                    "stage2_reward_min": stage2_rewards_original.min(),
                    "stage2_reward_mean": stage2_rewards_original.mean(),
                    "stage2_reward_max": stage2_rewards_original.max(),
                    "stage2_baseline_mean": stage2_baseline.mean(),
                    "stage2_std_mean": stage2_std.mean(),
                    "stage2_perfect_prediction_rate": (stage2_rewards_original == 0).float().mean(),  # Percentage of perfect predictions
                    "stage2_percent_zero_advantages": (stage2_advantages == 0).float().mean(),  # Zero advantages for stage 2
                    "combined_reward_min": all_rewards.min(),
                    "combined_reward_mean": all_rewards.mean(),
                    "combined_reward_max": all_rewards.max(),
                    "percent_zero_advantages": (all_advantages == 0).float().mean(),  # Keep combined metric for backward compatibility
                    "stage2_better_than_stage1_avg_rate": stage2_better_than_stage1_avg_rate,
                    # Majority cluster metrics
                    "stage1_majority_rate": majority_rate,
                    "stage2_same_as_majority_rate": stage2_same_as_majority_rate,
                    "stage2_better_than_majority_rate": stage2_better_than_majority_rate,
                    "stage1_avg_majority_size": avg_majority_size,
                })
                
                # Add binary reward metrics if enabled
                if master_config["pt_grpo"].get("use_binary_reward_for_stage2", False):
                    rollout_metrics["stage2_binary_reward_mean"] = stage2_rewards.mean()  # Binary reward rate
                
                # Add cluster overcome bonus metrics if enabled
                if use_stage2_cluster_bonus and 'bonus_count' in locals():
                    rollout_metrics["stage2_cluster_bonus_count"] = bonus_count
                    rollout_metrics["stage2_cluster_bonus_rate"] = bonus_count / len(stage2_rewards) if len(stage2_rewards) > 0 else 0.0
                    rollout_metrics["stage2_cluster_bonus_total"] = total_bonus_added

            # ============== Prepare Training Data ==============
            print("\n▶ Preparing training data...")
            with timer.time("data_processing"):
                # Prepare stage 1 training data
                for i, message_log in enumerate(stage1_repeated_batch["message_log"]):
                    for j, message in enumerate(message_log):
                        if message["role"] == "assistant":
                            message["token_loss_mask"] = torch.ones_like(message["token_ids"])
                        else:
                            message["token_loss_mask"] = torch.zeros_like(message["token_ids"])
                        if "generation_logprobs" not in message:
                            message["generation_logprobs"] = torch.zeros_like(
                                message["token_ids"], dtype=torch.float32
                            )
                        message["advantages"] = stage1_advantages[i].expand(message["token_ids"].shape)

                # Convert stage 1 to training data
                stage1_flat_messages, stage1_input_lengths = batched_message_log_to_flat_message(
                    stage1_repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    make_sequence_length_divisible_by=master_config["policy"]["make_sequence_length_divisible_by"],
                )

                stage1_train_data = BatchedDataDict[ClippedPGLossDataDict]({
                    "input_ids": stage1_flat_messages["token_ids"],
                    "input_lengths": stage1_input_lengths,
                    "advantages": stage1_flat_messages["advantages"],
                    "generation_logprobs": stage1_flat_messages["generation_logprobs"],
                    "token_mask": stage1_flat_messages["token_loss_mask"],
                    "sample_mask": stage1_repeated_batch["loss_multiplier"],
                })

                # Prepare stage 2 training data
                for i, message_log in enumerate(stage2_repeated_batch["message_log"]):
                    for j, message in enumerate(message_log):
                        if message["role"] == "assistant":
                            message["token_loss_mask"] = torch.ones_like(message["token_ids"])
                        else:
                            message["token_loss_mask"] = torch.zeros_like(message["token_ids"])
                        if "generation_logprobs" not in message:
                            message["generation_logprobs"] = torch.zeros_like(
                                message["token_ids"], dtype=torch.float32
                            )
                        message["advantages"] = stage2_advantages[i].expand(message["token_ids"].shape)

                # Convert stage 2 to training data
                stage2_flat_messages, stage2_input_lengths = batched_message_log_to_flat_message(
                    stage2_repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    make_sequence_length_divisible_by=master_config["policy"]["make_sequence_length_divisible_by"],
                )

                stage2_train_data = BatchedDataDict[ClippedPGLossDataDict]({
                    "input_ids": stage2_flat_messages["token_ids"],
                    "input_lengths": stage2_input_lengths,
                    "advantages": stage2_flat_messages["advantages"],
                    "generation_logprobs": stage2_flat_messages["generation_logprobs"],
                    "token_mask": stage2_flat_messages["token_loss_mask"],
                    "sample_mask": stage2_repeated_batch["loss_multiplier"],
                })

                # Combine and shuffle training data from both stages
                train_data = combine_and_shuffle_training_data(stage1_train_data, stage2_train_data)
                train_data.to("cpu")

            # ============== Policy Update ==============
            print("\n▶ Updating policy...")
            with timer.time("logprob_inference_prep"):
                policy.prepare_for_lp_inference()

            with timer.time("policy_and_reference_logprobs"):
                fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]
                reference_logprobs = policy.get_reference_policy_logprobs(train_data)["reference_logprobs"]
                train_data["prev_logprobs"] = fprop_logprobs
                train_data["reference_policy_logprobs"] = reference_logprobs

            with timer.time("training_prep"):
                policy.prepare_for_training()  # set model train and reload optim to GPU
                POLICY_GENERATION_STALE = True

            with timer.time("policy_training"):
                list_of_train_metrics = policy.train(train_data, loss_fn)

            is_last_step = step + 1 == max_num_steps

            # Run validation if it's a validation step
            if is_last_step or (val_period > 0 and (step + 1) % val_period == 0):
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(
                        policy,
                        policy_generation,
                        refit_buffer_size_gb,
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()
                val_metrics, validation_timings = validate(
                    policy_generation,
                    val_dataloader,
                    tokenizer,
                    val_task_to_env,
                    step=step + 1,
                    master_config=master_config,
                    logger=logger,
                )
                policy_generation.finish_generation()
                logger.log_metrics(validation_timings, step + 1, prefix="timing/validation")
                logger.log_metrics(val_metrics, step + 1, prefix="validation")

            ## Checkpointing
            consumed_samples += master_config["pt_grpo"]["num_prompts_per_step"]
            if master_config["checkpointing"]["enabled"] and (
                is_last_step
                or (step + 1) % master_config["checkpointing"]["save_period"] == 0
            ):
                policy.prepare_for_training()

                pt_grpo_save_state["step"] = step + 1
                pt_grpo_save_state["val_reward"] = val_metrics["accuracy"] if val_metrics else 0.0
                pt_grpo_save_state["consumed_samples"] = consumed_samples
                pt_grpo_save_state["optim_step"] = optim_step + len(list_of_train_metrics)
                with timer.time("checkpointing"):
                    print(f"  • Saving checkpoint for step {step + 1}...")
                    checkpoint_path = checkpointer.init_tmp_checkpoint(
                        step + 1, pt_grpo_save_state, master_config
                    )
                    policy.save_checkpoint(
                        weights_path=os.path.join(checkpoint_path, "policy", "weights"),
                        optimizer_path=os.path.join(checkpoint_path, "policy", "optimizer"),
                        tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
                    )
                    torch.save(
                        dataloader.state_dict(),
                        os.path.join(checkpoint_path, "train_dataloader.pt"),
                    )
                    checkpointer.finalize_checkpoint(checkpoint_path)
                policy.offload_after_refit()

        # ============== Logging ==============
        print("\n📊 Training Results:")
        print(f"  • Combined Avg Reward: {all_rewards.mean():.4f}")
        print(f"  • Stage 1 Avg Reward: {stage1_rewards.mean():.4f}")
        print(f"  • Stage 2 Avg Reward: {stage2_rewards_original.mean():.4f}")
        print(f"  • Stage 2 Better Than Stage 1 Avg Rate: {stage2_better_than_stage1_avg_rate:.2%}")
        if binary_reward_rate is not None:
            print(f"  • Binary Reward Rate: {binary_reward_rate:.2%}")
        print(f"  • Stage 1 Mean Gen Length: {rollout_metrics.get('stage1_mean_gen_tokens_per_sample', 0):.1f}")
        print(f"  • Stage 2 Mean Gen Length: {rollout_metrics.get('stage2_mean_gen_tokens_per_sample', 0):.1f}")

        # Log training data samples
        # Use stage1_flat_messages content (just like grpo.py)
        log_data = {"content": stage1_flat_messages["content"]}
        log_data["rewards"] = stage1_rewards.tolist()
        log_data["dataset_names"] = stage1_repeated_batch.get("dataset_names", ["default"] * len(stage1_rewards))
        
        # Add parallel thinking specific data
        pt_data = extract_parallel_thinking_log_data(
            batch=batch,
            stage1_repeated_batch=stage1_repeated_batch,
            stage1_rewards=stage1_rewards,
            stage2_repeated_batch=stage2_repeated_batch,
            stage2_rewards=stage2_rewards_original,  # Always use original rewards for logging
            num_generations_per_prompt=master_config["pt_grpo"]["num_generations_per_prompt"],
        )
        log_data.update(pt_data)
        
        # Log to JSONL with full data
        logger.log_batched_dict_as_jsonl(log_data, f"train_data_step{step}.jsonl")
        
        # Log table with parallel thinking data
        table = logger.log_batched_dict_as_table(log_data, prefix="train", step=step)

        rollout_metrics["table"] = table
        timing_metrics = timer.get_timing_metrics(reduction_op="sum")

        print("\n⏱️  Timing:")
        total_time = timing_metrics.get("total_step_time", 0)
        print(f"  • Total step time: {total_time:.2f}s")

        for k, v in sorted(timing_metrics.items(), key=lambda item: item[1], reverse=True):
            if k != "total_step_time":
                percent = (v / total_time * 100) if total_time > 0 else 0
                print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

        for i, train_step_metric in enumerate(list_of_train_metrics):
            train_step_metric["optim_step"] = optim_step + i + 1
            train_step_metric["outer_loop_step"] = step + 1
            logger.log_metrics(
                train_step_metric,
                train_step_metric["optim_step"],
                prefix="train",
            )

        logger.log_metrics(rollout_metrics, step + 1, prefix="train_rollout")
        logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

        timer.reset()
        step += 1
        optim_step += len(list_of_train_metrics)

        if step >= max_num_steps:
            break


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer,
    val_task_to_env: Dict[str, EnvironmentInterface],
    step: int,
    master_config: MasterConfig,
    logger: Optional[Logger] = None,
    num_repeats: Optional[int] = None,
    return_data_for_saving: bool = False,
    return_val_batch: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation")
        if return_val_batch:
            return {}, {}, [], None
        elif return_data_for_saving:
            return {}, {}, []
        else:
            return {}, {}

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"\n🔍 Running validation at step {step}...")

        # Get num_val_repeats from config or parameter
        if num_repeats is None:
            num_repeats = master_config["pt_grpo"].get("num_val_repeats", 1)
        
        total_rewards = []
        all_message_logs = []
        data_for_saving = []

        try:
            val_batch = next(iter(val_dataloader)).repeat_interleave(num_repeats)
        except StopIteration:
            print("  No validation data, skipping validation")
            if return_val_batch:
                return {}, {}, [], None
            elif return_data_for_saving:
                return {}, {}, []
            else:
                return {}, {}

        # Generate responses
        val_batch, gen_metrics = run_multi_turn_rollout(
            policy_generation,
            val_batch,
            tokenizer,
            val_task_to_env,
            max_seq_len=master_config["policy"]["max_total_sequence_length"],
            max_rollout_turns=master_config["pt_grpo"]["max_rollout_turns"],
            greedy=False,
        )

        # Apply environment post-processing for validation
        if master_config["pt_grpo"].get("skip_val_env_post_processing", False):
            val_env_metrics = {}
            print("    (Skipped - skip_val_env_post_processing is enabled)")
        else:
            val_batch, val_env_metrics = apply_environment_post_processing(
                val_batch, val_task_to_env, prefix="val_"
            )
        gen_metrics.update(val_env_metrics)

        # Collect message logs for later display
        to_env = [
            get_keys_from_message_log(val_batch["message_log"][i], ["role", "content"])
            for i in range(len(val_batch["message_log"]))
        ]
        all_message_logs.extend(to_env)
        total_rewards.extend(val_batch["total_reward"].tolist())

        if return_data_for_saving:
            # Transpose val_batch from batch-first to sample-first
            batch_size = val_batch.size
            repeat_idx_counter = defaultdict(int)
            for i in range(batch_size):
                sample_dict = {}
                for k, v in val_batch.items():
                    # hack to use the env stuff
                    if k == "message_log":
                        v = to_env

                    val = v[i]
                    if torch.is_tensor(val):
                        val = val.item()
                    sample_dict[k] = val

                    if k == "idx":
                        repeat_idx = repeat_idx_counter[val]
                        repeat_idx_counter[val] += 1

                sample_dict["eval_idx"] = f"{sample_dict['idx']}_{repeat_idx}"
                data_for_saving.append(sample_dict)

        # Log one example for each unique dataset
        unique_datasets = list(set(val_batch.get("dataset_names", ["default"])))
        table = None

        for dataset_name in unique_datasets:
            if "dataset_names" in val_batch:
                dataset_idx = val_batch["dataset_names"].index(dataset_name)
            else:
                dataset_idx = 0

            for interaction in val_batch["message_log"][dataset_idx]:
                if interaction["role"] == "user":
                    prompt = interaction["content"]
                elif interaction["role"] == "assistant":
                    response = interaction["content"]
                else:
                    environment = interaction["content"]

            reward = val_batch["total_reward"][dataset_idx].item()

            if logger is not None:
                table = logger.log_table_contents(
                    step,
                    prompt,
                    response,
                    environment,
                    reward,
                    dataset_name,
                    f"validation/{dataset_name}",
                )

        val_metrics = {
            "table": table,
        }
        val_metrics.update(gen_metrics)

        # Calculate dataset-specific pass@k metrics
        if "dataset_names" in val_batch and "idx" in val_batch:
            prompt_based_reward_dict = defaultdict(list)
            idx_dictionary = defaultdict(list)
            for dataset, r, idx in zip(
                val_batch["dataset_names"], val_batch["total_reward"], val_batch["idx"]
            ):
                prompt_based_reward_dict[dataset].append(r)
                idx_dictionary[dataset].append(idx)

            for dataset, rewards in prompt_based_reward_dict.items():
                rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32).view(
                    -1, num_repeats
                )
                val_metrics[f"{dataset}/pass_at_{num_repeats}"] = (
                    (rewards_tensor > 0).any(-1).float().mean()
                )

        # Print message log samples
        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples=min(
                    master_config["logger"]["num_val_samples_to_print"],
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...")

        # Calculate validation metrics
        val_metrics["accuracy"] = val_batch["total_reward"].mean().item()
        val_metrics["mean_reward"] = val_batch["total_reward"].mean().item()
        val_metrics["mean_length"] = gen_metrics.get("mean_gen_tokens_per_sample", 0)
        val_metrics["num_samples"] = len(val_batch["total_reward"])

        print(f"  ✓ Validation complete: mean_reward={val_metrics['mean_reward']:.4f}, accuracy={val_metrics['accuracy']:.4f}")

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()
    
    if return_val_batch:
        # add token loss mask
        for i, message_log in enumerate(val_batch["message_log"]):
            for j, message in enumerate(message_log):
                if message["role"] == "assistant":
                    message["token_loss_mask"] = torch.ones_like(message["token_ids"])
                else:
                    message["token_loss_mask"] = torch.zeros_like(message["token_ids"])

        flat_messages, input_lengths = batched_message_log_to_flat_message(
            val_batch["message_log"],
            pad_value_dict={"token_ids": tokenizer.pad_token_id},
            make_sequence_length_divisible_by=master_config["policy"][
                "make_sequence_length_divisible_by"
            ],
        )

        # Create validation data
        val_data = BatchedDataDict[ClippedPGLossDataDict](
            {
                "input_ids": flat_messages["token_ids"],
                "input_lengths": input_lengths,
                "token_mask": flat_messages["token_loss_mask"],
            }
        )
        val_data.to("cpu")
        return val_metrics, timing_metrics, data_for_saving, val_data
    elif return_data_for_saving:
        return val_metrics, timing_metrics, data_for_saving
    else:
        return val_metrics, timing_metrics 