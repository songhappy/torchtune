#!/bin/bash
#PBS -N guoqiong_q_70b        
#PBS -q prod
#PBS -l walltime=2:30:00
#PBS -l select=1
#PBS -l filesystems=flare
#PBS -A Intel-Aurora
#PBS -o prof_70b_qlora.log  
#PBS -e prof_70b_qlora.log  

cd $PBS_O_WORKDIR

source ~/env.sh
source /home/songhappy/miniforge3/etc/profile.d/conda.sh
conda activate guoqiong-pt
# export TORCH_CPP_LOG_LEVEL=WARNING
# export TORCH_CPP_LOG_LEVEL=INFO
# nohup xpu-smi dump -m 2,18 -i 1 > /home/songhappy/git/torchtune/output/llama3_3/xpu_mem_log_qlora.txt 2>&1 &
torchrun --nproc_per_node 2 ../codelearn/python/llm_finetune/train_fsdp2.py

start_sec=$(date +%s)
echo "***********start qlora t8"
tune run  --nproc_per_node 8 lora_finetune_distributed --config /home/songhappy/git/torchtune/recipes/configs/llama3_3/70B_lora.yaml \
    device=xpu   seed=123 dataset.packed=True log_level=DEBUG\
    model._component_=torchtune.models.llama3_1.qlora_llama3_1_70b \
    tokenizer.path=/home/songhappy/models/Llama-3.3-70B-Instruct/original/tokenizer.model \
    checkpointer.checkpoint_dir=/home/songhappy/models/Llama-3.3-70B-Instruct/ \
    output_dir=/home/songhappy/git/torchtune/output/llama3_3/70b_qlora_t8 \
    max_steps_per_epoch=10 tokenizer.max_seq_len=512 batch_size=2 profiler.enabled=False \
    > 70b_qlora_t8
    
end_sec=$(date +%s)
duration_min=$((duration_sec / 60))
echo "Duration:       $duration_min minutes"

