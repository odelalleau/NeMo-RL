#!/bin/bash
#SBATCH -N 1 --gpus-per-node=8 --ntasks-per-node 1 -A llmservice_modelalignment_ppo -p interactive --job-name grpo_genrm_hs3_3B -t 04:00:00 
#SBATCH --mem=0 --dependency singleton



export NCCL_ALGO=Tree

set -x

GPFS="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL"
CONTAINER="/lustre/fsw/portfolios/llmservice/users/yizhuj/nemorl/containers/anyscale+ray+2.43.0-py312-cu125_uv.sqsh"
export HF_HOME=/lustre/fsw/portfolios/llmservice/users/yizhuj/hf_cache




#MODEL="nvidia/Llama-3_3-Nemotron-Super-49B-v1"
#MODEL_NAME="Llama-3_3-Nemotron-Super-49B"
FSDP2=True
MODEL="meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME="llama3.2_3B"

ACT_CKPT=True
CPU_OFFLOAD=False
TP=1
project_name="yizhu_rlhf"
lr=2e-6
temp=1
grpo_bs=64
prompts_per_step=128
rollouts_per_prompt=8
kl=0.001

NAME="grpo_hs3_16K_step240_clip_max_0.28_${MODEL_NAME}_lr_${lr}_temp_${temp}_kl_${kl}_grpo_bs_${grpo_bs}_rollout_${rollouts_per_prompt}_num_prompts_${prompts_per_step}"

RESULTS_DIR="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL/results/${NAME}"
mkdir -p $RESULTS_DIR

MOUNTS="--container-mounts=${GPFS}:${GPFS},/lustre:/lustre"

ACTOR_LOG_DIR="${RESULTS_DIR}/${SLURM_JOB_ID}"

PPO_ERRFILE="${ACTOR_LOG_DIR}/%j_%t.err"
PPO_OUTFILE="${ACTOR_LOG_DIR}/%j_%t.log"

mkdir -p $ACTOR_LOG_DIR

read -r -d '' cmd_ppo <<EOF
cd ${GPFS} \
&& ulimit -c 0 \
&& uv run examples/run_grpo_genrm.py \
    ++logger.wandb.name=${NAME} \
    ++logger.wandb_enabled=True \
    logger.wandb.project=${project_name} \
    ++checkpointing.checkpoint_dir=${RESULTS_DIR} \
    ++policy.model_name=${MODEL} \
    ++policy.make_sequence_length_divisible_by=8 \
    ++policy.generation.vllm_cfg.tensor_parallel_size=${TP} \
    ++cluster.num_nodes=${SLURM_NNODES} \
    policy.dtensor_cfg.enabled=${FSDP2} \
    policy.dtensor_cfg.tensor_parallel_size=${TP} \
    policy.dtensor_cfg.activation_checkpointing=${ACT_CKPT} \
    policy.dtensor_cfg.cpu_offload=${CPU_OFFLOAD} \
    ++cluster.gpus_per_node=8 \
    grpo.num_prompts_per_step=${prompts_per_step} \
    grpo.num_generations_per_prompt=${rollouts_per_prompt} \
    grpo.val_period=1 \
    grpo.max_val_samples=64 \
    grpo.val_batch_size=64 \
    data.train_data_path="/lustre/fsw/portfolios/llmservice/users/yizhuj/datasets/hs3_genrm/train_data.jsonl" \
    data.val_data_path="/lustre/fsw/portfolios/llmservice/users/yizhuj/datasets/hs3_genrm/val_data.jsonl" \
    loss_fn.reference_policy_kl_penalty=${kl} \
    loss_fn.use_on_policy_kl_approximation=False \
    loss_fn.use_importance_sampling_correction=False \
    checkpointing.keep_top_k=10 \
    checkpointing.save_period=10 \
    loss_fn.ratio_clip_min=0.2 \
    loss_fn.ratio_clip_max=0.28 \
    policy.train_global_batch_size=${grpo_bs} \
    policy.train_micro_batch_size=1 \
    policy.generation_batch_size=32 \
    policy.logprob_batch_size=1 \
    policy.max_total_sequence_length=8192 \
    policy.optimizer.kwargs.lr=${lr} \
    policy.optimizer.kwargs.weight_decay=0 \
    policy.generation.temperature=${temp} \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.6 \
    data.dataset_name="hs3" \
    env.genrm.num_workers=1
EOF

srun -o $PPO_OUTFILE -e $PPO_ERRFILE --container-image=${CONTAINER} $MOUNTS bash -c "${cmd_ppo}"
