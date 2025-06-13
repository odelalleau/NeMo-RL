#!/bin/bash

set -e

# Input arguments
CHECKPOINT_DIR="$1"
THRESHOLD="${2:-0}"
OUTPUT_BASE_DIR="${3:-${CHECKPOINT_DIR}/HF}"

# Container configuration
GPFS="/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL"
CONTAINER="/lustre/fsw/portfolios/llmservice/users/yizhuj/nemorl/containers/anyscale+ray+2.43.0-py312-cu125_uv.sqsh"

# SLURM configuration
ACCOUNT="llmservice_modelalignment_ppo"
PARTITION="interactive"
TIME_LIMIT="04:00:00"
NODES=1
CPUS_PER_TASK=8
GPUS_PER_NODE=8



# Create output directories
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "$OUTPUT_BASE_DIR/logs"

echo "Submitting DCP to HuggingFace conversion jobs"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Step threshold: $THRESHOLD"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "Container image: $CONTAINER"
# ... (rest of echo statements)
echo "----------------------------------------"

# Track submitted jobs
SUBMITTED_JOBS=()
SKIPPED_STEPS=()

# Function to submit a single conversion job
submit_conversion_job() {
    local step_dir="$1"
    local step_num="$2"
    
    local job_name="dcp2hf_step${step_num}"
    local log_file="$OUTPUT_BASE_DIR/logs/step_${step_num}.out"
    local err_file="$OUTPUT_BASE_DIR/logs/step_${step_num}.err"
    
    # Set up container mounts
    local mounts="--container-mounts=${GPFS}:${GPFS},/lustre:/lustre"
    
    # Define paths as they will appear INSIDE the container
    local container_config_path="${CHECKPOINT_DIR}/$(basename "$step_dir")/config.yaml"
    local container_dcp_path="${CHECKPOINT_DIR}/$(basename "$step_dir")/policy/weights"
    local container_hf_path="${OUTPUT_BASE_DIR}/step_${step_num}"
    local container_script_path="${GPFS}/examples/convert_dcp_to_hf.py"

    # Submit SLURM job.
    local job_output
    job_output=$(sbatch <<EOF
#!/bin/bash
#SBATCH -A $ACCOUNT
#SBATCH -J $job_name
#SBATCH -p $PARTITION
#SBATCH -t $TIME_LIMIT
#SBATCH -N $NODES
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --gpus-per-node=$GPUS_PER_NODE
#SBATCH -o $log_file
#SBATCH -e $err_file

export HF_HOME=/lustre/fsw/portfolios/llmservice/users/yizhuj/hf_cache

srun --container-image="$CONTAINER" $mounts bash -c '
    # Enable error-exit and execution tracing inside the container
    set -e
    set -x

    # Change to the NeMo-RL directory (this is crucial!)
    cd "'"$GPFS"'" \
    && ulimit -c 0 \
    && uv run python examples/convert_dcp_to_hf.py --config='"$container_config_path"' --dcp-ckpt-path='"$container_dcp_path"' --hf-ckpt-path='"$container_hf_path"' 

    echo "Conversion successful for step_'"$step_num"'."
'
EOF
)
    
    local job_id=$(echo "$job_output" | grep -oE '[0-9]+' | tail -1)
    echo "  📤 Submitted job $job_id for step_$step_num"
    SUBMITTED_JOBS+=("$job_id:step_$step_num")
}

# Process each step directory
for step_dir in "$CHECKPOINT_DIR"/step_*; do
    if [ -d "$step_dir" ]; then
        # Extract step number from directory name (anchor to end of string)
        step_basename=$(basename "$step_dir")
        step_num=$(echo "$step_basename" | grep -Eo '[0-9]+$')
        
        if [ ! -z "$step_num" ] && [ "$step_num" -gt "$THRESHOLD" ]; then
            echo "Processing directory: $step_dir"
            echo "Step number: $step_num"
            
            # Define host paths
            config_path="$step_dir/config.yaml"
            dcp_weights_path="$step_dir/policy/weights"
            hf_output_path="$OUTPUT_BASE_DIR/step_$step_num"
            
            # Validate required files exist
            if [ ! -f "$config_path" ]; then
                echo "  ❌ Error: config.yaml not found at $config_path"
                SKIPPED_STEPS+=("step_$step_num (missing config.yaml)")
                continue
            fi
            
            if [ ! -d "$dcp_weights_path" ]; then
                echo "  ❌ Error: DCP weights directory not found at $dcp_weights_path"
                SKIPPED_STEPS+=("step_$step_num (missing weights)")
                continue
            fi
            
            # Check if output already exists
            if [ -d "$hf_output_path" ]; then
                echo "  ⚠️  Output directory already exists: $hf_output_path"
                read -p "  Overwrite? (y/N): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    echo "  ⏭️  Skipping step_$step_num"
                    SKIPPED_STEPS+=("step_$step_num (already exists, user skipped)")
                    continue
                else
                    echo "  ♻️  Removing existing output directory."
                    rm -rf "$hf_output_path"
                fi
            fi
            
            # Submit conversion job
            submit_conversion_job "$step_dir" "$step_num"
            
        else
            if [ ! -z "$step_num" ]; then
                echo "Skipping $step_basename (step $step_num <= threshold $THRESHOLD)"
                SKIPPED_STEPS+=("$step_basename (below threshold)")
            fi
        fi
    fi
done
