#!/bin/bash
#SBATCH -N 1 --gpus-per-node=8 --ntasks-per-node 1 -A llmservice_modelalignment_ppo -p interactive --job-name eval_genrm_custom -t 04:00:00 

export NCCL_ALGO=Tree
set -x

# Configuration
GPFS="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL"
CONTAINER="/lustre/fsw/portfolios/llmservice/users/yizhuj/nemorl/containers/anyscale+ray+2.43.0-py312-cu125_uv.sqsh"
export HF_HOME=/lustre/fsw/portfolios/llmservice/users/yizhuj/hf_cache


# Base directory for your specific checkpoint structure
BASE_DIR="${1:-/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL/results/grpo_hs3_16K_step240_clip_max_0.28_llama3.2_3B_lr_2e-6_temp_1_kl_0.001_grpo_bs_64_rollout_8_num_prompts_128}"
HF_DIR="${BASE_DIR}/HF"  # Your checkpoints are under HF/step_X
RESULTS_DIR="${2:-/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL/outputs/genrm_eval_results/grpo_hs3_16K_step240_clip_max_0.28_llama3.2_3B_lr_2e-6_temp_1_kl_0.001_grpo_bs_64_rollout_8_num_prompts_128}"

# Create results directory
mkdir -p $RESULTS_DIR

# Job info
JOB_LOG_DIR=${RESULTS_DIR}
mkdir -p $JOB_LOG_DIR

MOUNTS="--container-mounts=${GPFS}:${GPFS},/lustre:/lustre"

# Create a summary file
SUMMARY_FILE="${JOB_LOG_DIR}/evaluation_summary.txt"
echo "GenRM Evaluation on JudgeBench Dataset" > $SUMMARY_FILE
echo "======================================" >> $SUMMARY_FILE
echo "Base Directory: $BASE_DIR" >> $SUMMARY_FILE
echo "HF Directory: $HF_DIR" >> $SUMMARY_FILE
echo "Start Time: $(date)" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Function to extract accuracy from JSON output
extract_accuracy() {
    local json_file=$1
    if [ -f "$json_file" ]; then
        accuracy=$(python3 -c "
import json
with open('$json_file', 'r') as f:
    results = json.load(f)
    
correct = 0
total = 0
for r in results:
    if 'predicted_ranking' in r and r['metadata'].get('preference') is not None:
        total += 1
        pred = r['predicted_ranking']
        true_pref = r['metadata']['preference']
        if (true_pref <= 3 and pred <= 3) or (true_pref > 3 and pred > 3):
            correct += 1

if total > 0:
    accuracy = correct / total
    print(f'{accuracy:.4f}')
else:
    print('N/A')
" 2>/dev/null)
        echo "$accuracy"
    else
        echo "N/A"
    fi
}

# Find all step directories under HF/ and sort them numerically
echo "Finding checkpoints in $HF_DIR..." | tee -a $SUMMARY_FILE
if [ -d "$HF_DIR" ]; then
    STEP_DIRS=$(find $HF_DIR -type d -name "step_*" | sort -V)
else
    echo "HF directory not found. Looking for step directories in base directory..." | tee -a $SUMMARY_FILE
    STEP_DIRS=$(find $BASE_DIR -type d -name "step_*" | sort -V)
fi

if [ -z "$STEP_DIRS" ]; then
    echo "No checkpoint directories found!" | tee -a $SUMMARY_FILE
    exit 1
fi

echo "Found checkpoints:" | tee -a $SUMMARY_FILE
echo "$STEP_DIRS" | tee -a $SUMMARY_FILE
echo "" | tee -a $SUMMARY_FILE

# Evaluate each checkpoint
for STEP_DIR in $STEP_DIRS; do
    STEP_NAME=$(basename $STEP_DIR)
    echo "----------------------------------------" | tee -a $SUMMARY_FILE
    echo "Evaluating checkpoint: $STEP_NAME" | tee -a $SUMMARY_FILE
    echo "Path: $STEP_DIR" | tee -a $SUMMARY_FILE
    
    # The model path is the step directory itself in your case
    MODEL_PATH="$STEP_DIR"
    
    # Define output file for this step
    OUTPUT_FILE="${JOB_LOG_DIR}/${STEP_NAME}_judgebench_results.json"
    LOG_FILE="${JOB_LOG_DIR}/${STEP_NAME}_eval.log"
    ERR_FILE="${JOB_LOG_DIR}/${STEP_NAME}_eval.err"
    
    # Run evaluation
    read -r -d '' cmd_eval <<EOF
cd ${GPFS} \
&& ulimit -c 0 \
&& uv run python examples/run_eval_genrm.py \
    --dataset judgebench \
    ++generation.model_name=${MODEL_PATH} \
    ++eval.output_file=${OUTPUT_FILE} \
    ++eval.batch_size=256 \
    ++generation.vllm_cfg.tensor_parallel_size=1 \
    ++generation.vllm_cfg.gpu_memory_utilization=0.7 \
    ++cluster.gpus_per_node=1 \
    ++cluster.num_nodes=1
EOF

    echo "Running evaluation for $STEP_NAME..." | tee -a $SUMMARY_FILE
    echo "Output will be saved to: $OUTPUT_FILE" | tee -a $SUMMARY_FILE
    
    # Execute the evaluation
    srun -o $LOG_FILE -e $ERR_FILE --container-image=${CONTAINER}  -A llmservice_modelalignment_ppo -p interactive --job-name eval_genrm_custom  -N 1 --gpus-per-node=1 --ntasks-per-node 1  $MOUNTS bash -c "${cmd_eval}"
    
    # Check if evaluation was successful
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed successfully" | tee -a $SUMMARY_FILE
        
        # Extract and display accuracy
        ACCURACY=$(extract_accuracy "$OUTPUT_FILE")
        echo "Accuracy: $ACCURACY" | tee -a $SUMMARY_FILE
        
        # Also save a formatted result
        echo "${STEP_NAME}: Accuracy = $ACCURACY" >> "${JOB_LOG_DIR}/accuracy_summary.txt"
    else
        echo "✗ Evaluation failed for $STEP_NAME" | tee -a $SUMMARY_FILE
        echo "Check error log: $ERR_FILE" | tee -a $SUMMARY_FILE
    fi
    
    echo "" | tee -a $SUMMARY_FILE
done

echo "======================================" | tee -a $SUMMARY_FILE
echo "Evaluation Complete!" | tee -a $SUMMARY_FILE
echo "End Time: $(date)" | tee -a $SUMMARY_FILE
echo "" | tee -a $SUMMARY_FILE
echo "Results saved in: $JOB_LOG_DIR" | tee -a $SUMMARY_FILE

# Display final accuracy summary
if [ -f "${JOB_LOG_DIR}/accuracy_summary.txt" ]; then
    echo "" | tee -a $SUMMARY_FILE
    echo "Accuracy Summary:" | tee -a $SUMMARY_FILE
    echo "----------------" | tee -a $SUMMARY_FILE
    cat "${JOB_LOG_DIR}/accuracy_summary.txt" | tee -a $SUMMARY_FILE
    
    # Create a sorted version by accuracy
    echo "" | tee -a $SUMMARY_FILE
    echo "Sorted by Accuracy (Best to Worst):" | tee -a $SUMMARY_FILE
    echo "-----------------------------------" | tee -a $SUMMARY_FILE
    sort -t'=' -k2 -rn "${JOB_LOG_DIR}/accuracy_summary.txt" | tee -a $SUMMARY_FILE
fi