#!/bin/bash
#PBS -N guoqiong_job         
#PBS -q prod
#PBS -l walltime=2:30:00
#PBS -l select=1
#PBS -l filesystems=flare
#PBS -A Intel-Aurora
#PBS -o prof_70b_full.log  
#PBS -e prof_70b_full.log  

cd $PBS_O_WORKDIR

source ~/env.sh
source /home/songhappy/miniforge3/etc/profile.d/conda.sh
conda activate guoqiong-pt-dis
# export TORCH_CPP_LOG_LEVEL=WARNING
# export TORCH_CPP_LOG_LEVEL=INFO
# nohup xpu-smi dump -m 2,18 -i 1 > /home/songhappy/git/torchtune/output/llama3_3/xpu_mem_log.txt 2>&1 &


# start_sec=$(date +%s)
# echo "***********start full t12"
# tune run  --nproc_per_node 12 full_finetune_distributed --config /home/songhappy/git/torchtune/recipes/configs/llama3_3/70B_lora.yaml \
#     device=xpu   seed=123 dataset.packed=True \
#     tokenizer.path=/home/songhappy/models/Llama-3.3-70B-Instruct/original/tokenizer.model \
#     checkpointer.checkpoint_dir=/home/songhappy/models/Llama-3.3-70B-Instruct/ \
#     output_dir=/home/songhappy/git/torchtune/output/llama3_3/70b_full_t12 \
#     max_steps_per_epoch=10 tokenizer.max_seq_len=512 batch_size=2 profiler.enabled=False \
#     > 70b_full_t12
    
# end_sec=$(date +%s)
# duration_min=$((duration_sec / 60))
# echo "Duration:       $duration_min minutes"


# echo "***********start full t8"
# tune run  --nproc_per_node 8 full_finetune_distributed --config /home/songhappy/git/torchtune/recipes/configs/llama3_3/70B_full.yaml \
#     device=xpu   seed=123 dataset.packed=True \
#     tokenizer.path=/home/songhappy/models/Llama-3.3-70B-Instruct/original/tokenizer.model \
#     checkpointer.checkpoint_dir=/home/songhappy/models/Llama-3.3-70B-Instruct/ \
#     output_dir=/home/songhappy/git/torchtune/output/llama3_3/70B_full_8t \
#     max_steps_per_epoch=10 tokenizer.max_seq_len=512 batch_size=2 profiler.enabled=False \
#     > 70b_full_t8