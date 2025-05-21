#!/bin/bash
#PBS -N gs_lora_8b_seq       
#PBS -q debug
#PBS -l walltime=1:00:00
#PBS -l select=1
#PBS -l filesystems=flare
#PBS -A Intel-Aurora
#PBS -o output/prof_8b_lora.log  
#PBS -e output/prof_8b_lora.log    

cd $PBS_O_WORKDIR

# === Environment Setup ===
source ~/env.sh
source /home/songhappy/miniforge3/etc/profile.d/conda.sh
conda activate guoqiong-pt

# export TORCH_CPP_LOG_LEVEL=INFO
export PYTHONUNBUFFERED=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

module load xpu-smi

# === Common Model Config ===
MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
TOKENIZER_PATH="/home/songhappy/models/${MODEL_NAME}/original/tokenizer.model"
CHECKPOINT_DIR="/home/songhappy/models/${MODEL_NAME}"
MAX_SEQ_LEN=512
BATCH_SIZE=2

# -----------------------
# 1. Single-device LoRA
# -----------------------
CONFIG="/home/songhappy/git/torchtune/recipes/configs/llama3_1/8B_lora_single_device.yaml"
OUTPUT_DIR="/home/songhappy/git/torchtune/output/llama3_1/8b_lora_sd"
LOG_TRAIN=/home/songhappy/git/torchtune/output/train_8b_lora_sd.log
LOG_MEMO=/home/songhappy/git/torchtune/output/xpu_memo_8b_lora_sd.log

mkdir -p "$OUTPUT_DIR"
echo "*********** Starting Single-Device LoRA ***********"
start_sd=$(date +%s)

echo "Launching xpu-smi for single-device run..."
nohup xpu-smi dump -m 2,18 -i 1 > "$LOG_MEMO" 2>&1 &

tune run lora_finetune_single_device \
    --config "$CONFIG" \
    device=xpu seed=123 dataset.packed=True log_level=DEBUG \
    tokenizer.path="$TOKENIZER_PATH" \
    checkpointer.checkpoint_dir="$CHECKPOINT_DIR" \
    output_dir="$OUTPUT_DIR" \
    max_steps_per_epoch=10 tokenizer.max_seq_len=$MAX_SEQ_LEN batch_size=$BATCH_SIZE profiler.enabled=False \
    > "$LOG_TRAIN" 2>&1

end_sd=$(date +%s)
dur_sd=$((end_sd - start_sd))
printf "Single-device duration: %02d:%02d:%02d\n" $((dur_sd/3600)) $((dur_sd%3600/60)) $((dur_sd%60))

# -----------------------------
# 2. Two-tile LoRA (DDP)
# -----------------------------
CONFIG="/home/songhappy/git/torchtune/recipes/configs/llama3_1/8B_lora.yaml"
OUTPUT_DIR="/home/songhappy/git/torchtune/output/llama3_1/8b_lora_t2"
LOG_TRAIN=/home/songhappy/git/torchtune/output/train_8b_lora_t2.log
LOG_MEMO=/home/songhappy/git/torchtune/output/xpu_memo_8b_lora_t2.log

mkdir -p "$OUTPUT_DIR"
echo "*********** Starting Two-Tile LoRA ***********"
start_t2=$(date +%s)

echo "Launching xpu-smi for two-tile run..."
nohup xpu-smi dump -m 2,18 -i 1 > "$LOG_MEMO" 2>&1 &

tune run --nproc_per_node=2 lora_finetune_distributed \
    --config "$CONFIG" \
    device=xpu seed=123 dataset.packed=True log_level=DEBUG \
    tokenizer.path="$TOKENIZER_PATH" \
    checkpointer.checkpoint_dir="$CHECKPOINT_DIR" \
    output_dir="$OUTPUT_DIR" \
    max_steps_per_epoch=10 tokenizer.max_seq_len=$MAX_SEQ_LEN batch_size=$BATCH_SIZE profiler.enabled=False \
    > "$LOG_TRAIN" 2>&1

end_t2=$(date +%s)
dur_t2=$((end_t2 - start_t2))
printf "Two-tile duration: %02d:%02d:%02d\n" $((dur_t2/3600)) $((dur_t2%3600/60)) $((dur_t2%60))
