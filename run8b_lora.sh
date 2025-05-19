#!/bin/bash
#PBS -N guoqiong_lora_8b        
#PBS -q debug
#PBS -l walltime=30:00
#PBS -l select=1
#PBS -l filesystems=flare
#PBS -A Intel-Aurora
#PBS -o prof_8b_lora.log  
#PBS -e prof_8b_lora.log  

cd $PBS_O_WORKDIR

source ~/env.sh
source /home/songhappy/miniforge3/etc/profile.d/conda.sh
conda activate guoqiong-pt
# export TORCH_CPP_LOG_LEVEL=WARNING
export TORCH_CPP_LOG_LEVEL=INFO
# nohup xpu-smi dump -m 2,18 -i 1 > /home/songhappy/git/torchtune/output/llama3_3/xpu_mem_log.txt 2>&1 &

start_sec=$(date +%s)
echo "***********start lora sd"
tune run lora_finetune_single_device --config /home/songhappy/git/torchtune/recipes/configs/llama3_1/8B_lora_single_device.yaml \
    device=xpu   seed=123 dataset.packed=True log_level=DEBUG\
    tokenizer.path=/home/songhappy/models/Meta-Llama-3.1-8B-Instruct/original/tokenizer.model \
    checkpointer.checkpoint_dir=/home/songhappy/models/Meta-Llama-3.1-8B-Instruct/ \
    output_dir=/home/songhappy/git/torchtune/output/llama3_1/8b_lora_sd \
    max_steps_per_epoch=10 tokenizer.max_seq_len=512 batch_size=2 profiler.enabled=False \
    > 8b_lora_sd
end_sec=$(date +%s)
duration_sec=$((end_sec - start_sec)) 
printf "Duration:       %02d:%02d:%02d\n" $((duration_sec/3600)) $((duration_sec%3600/60)) $((duration_sec%60))


# start_sec=$(date +%s)
# echo "***********start lora t2"
# tune run  --nproc_per_node 2 lora_finetune_distributed --config /home/songhappy/git/torchtune/recipes/configs/llama3_1/8B_lora.yaml \
#     device=xpu   seed=123 dataset.packed=True log_level=DEBUG\
#     tokenizer.path=/home/songhappy/models/Meta-Llama-3.1-8B-Instruct/original/tokenizer.model \
#     checkpointer.checkpoint_dir=/home/songhappy/models/Meta-Llama-3.1-8B-Instruct/ \
#     output_dir=/home/songhappy/git/torchtune/output/llama3_1/8b_lora_t2 \
#     max_steps_per_epoch=10 tokenizer.max_seq_len=512 batch_size=2 profiler.enabled=False \
#     > 8b_lora_t2
# end_sec=$(date +%s)
# duration_sec=$((end_sec - start_sec)) 
# printf "Duration:       %02d:%02d:%02d\n" $((duration_sec/3600)) $((duration_sec%3600/60)) $((duration_sec%60))

# start_sec=$(date +%s)
# echo "***********start lora t4"
# tune run  --nproc_per_node 4 lora_finetune_distributed --config /home/songhappy/git/torchtune/recipes/configs/llama3_1/8B_lora.yaml \
#     device=xpu   seed=123 dataset.packed=True log_level=DEBUG\
#     tokenizer.path=/home/songhappy/models/Meta-Llama-3.1-8B-Instruct/original/tokenizer.model \
#     checkpointer.checkpoint_dir=/home/songhappy/models/Meta-Llama-3.1-8B-Instruct/ \
#     output_dir=/home/songhappy/git/torchtune/output/llama3_1/8b_lora_t4 \
#     max_steps_per_epoch=10 tokenizer.max_seq_len=512 batch_size=2 profiler.enabled=False \
#     > 8b_lora_t4
# end_sec=$(date +%s)
# duration_sec=$((end_sec - start_sec))   
# printf "Duration:       %02d:%02d:%02d\n" $((duration_sec/3600)) $((duration_sec%3600/60)) $((duration_sec%60))

