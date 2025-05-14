#!/bin/bash
#PBS -N guoqiong_q_405b        
#PBS -q prod
#PBS -l walltime=2:30:00
#PBS -l select=1
#PBS -l filesystems=flare
#PBS -A Intel-Aurora
#PBS -o prof_405b_qlora.log  
#PBS -e prof_405b_qlora.log  

cd $PBS_O_WORKDIR

source ~/env.sh
source /home/songhappy/miniforge3/etc/profile.d/conda.sh
conda activate guoqiong-pt
# export TORCH_CPP_LOG_LEVEL=WARNING
# export TORCH_CPP_LOG_LEVEL=INFO
# nohup xpu-smi dump -m 2,18 -i 1 > /home/songhappy/git/torchtune/output/llama3_3/xpu_mem_log.txt 2>&1 &
# torchrun --nproc_per_node 2 ../codelearn/python/llm_finetune/train_fsdp2.py

start_sec=$(date +%s)
echo "***********start qlora t8"
tune run  --nproc_per_node 8 lora_finetune_distributed --config /home/songhappy/git/torchtune/recipes/configs/llama3_1/405B_qlora.yaml \
    device=xpu   seed=123 dataset.packed=True log_level=DEBUG\
    tokenizer.path=/flare/Aurora_deployment/meta-llama/405B/Meta-Llama-3.1-405B-Instruct/original/mp8/tokenizer.model \
    checkpointer.checkpoint_dir=/flare/Aurora_deployment/meta-llama/405B/Meta-Llama-3.1-405B-Instruct/ \
    output_dir=/home/songhappy/git/torchtune/output/llama3_1/405b_qlora_t8 \
    max_steps_per_epoch=10 tokenizer.max_seq_len=512 batch_size=2 profiler.enabled=False \
    > 405b_qlora_t8
    
end_sec=$(date +%s)
duration_min=$((duration_sec / 60))
echo "Duration:       $duration_min minutes"

