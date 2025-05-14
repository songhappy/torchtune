#!/bin/bash
#PBS -N guoqiong_qlora_8b         
#PBS -q prod
#PBS -l walltime=2:30:00
#PBS -l select=1
#PBS -l filesystems=flare
#PBS -A Intel-Aurora
#PBS -o prof_8b_qlora.log  
#PBS -e prof_8b_qlora.log  

cd $PBS_O_WORKDIR

source ~/env.sh
source /home/songhappy/miniforge3/etc/profile.d/conda.sh
conda activate guoqiong-pt-dis
# export TORCH_CPP_LOG_LEVEL=WARNING
# export TORCH_CPP_LOG_LEVEL=INFO
# nohup xpu-smi dump -m 2,18 -i 1 > /home/songhappy/git/torchtune/output/llama3_3/xpu_mem_log.txt 2>&1 &

start_sec=$(date +%s)
echo "***********start qlora sd"
tune run lora_finetune_single_device --config /home/songhappy/git/torchtune/recipes/configs/llama3_1/8B_lora_single_device.yaml \
    device=xpu   seed=123 dataset.packed=True log_level=DEBUG\
    tokenizer.path=/home/songhappy/models/Meta-Llama-3.1-8B-Instruct/original/tokenizer.model \
    checkpointer.checkpoint_dir=/home/songhappy/models/Meta-Llama-3.1-8B-Instruct/ \
    model._component_=torchtune.models.llama3_1.qlora_llama3_1_8b \
    output_dir=/home/songhappy/git/torchtune/output/llama3_1/8b_qlora_sd \
    max_steps_per_epoch=10 tokenizer.max_seq_len=512 batch_size=2 profiler.enabled=False \
    > 8b_qlora_sd
end_sec=$(date +%s)
duration_sec=$((end_sec - start_sec)) 
printf "Duration:       %02d:%02d:%02d\n" $((duration_sec/3600)) $((duration_sec%3600/60)) $((duration_sec%60))


start_sec=$(date +%s)
echo "***********start qlora t2"
tune run  --nproc_per_node 2 lora_finetune_distributed --config /home/songhappy/git/torchtune/recipes/configs/llama3_1/8B_lora.yaml \
    device=xpu   seed=123 dataset.packed=True \
    tokenizer.path=/home/songhappy/models/Meta-Llama-3.1-8B-Instruct/original/tokenizer.model \
    checkpointer.checkpoint_dir=/home/songhappy/models/Meta-Llama-3.1-8B-Instruct/ \
    model._component_=torchtune.models.llama3_1.qlora_llama3_1_8b \
    output_dir=/home/songhappy/git/torchtune/output/llama3_1/8b_qlora_t2 \
    max_steps_per_epoch=10 tokenizer.max_seq_len=512 batch_size=2 profiler.enabled=False \
    > 8b_qlora_t2
end_sec=$(date +%s)
duration_sec=$((end_sec - start_sec)) 
printf "Duration:       %02d:%02d:%02d\n" $((duration_sec/3600)) $((duration_sec%3600/60)) $((duration_sec%60))


start_sec=$(date +%s)
echo "***********start qlora t4"
tune run  --nproc_per_node 4 lora_finetune_distributed --config /home/songhappy/git/torchtune/recipes/configs/llama3_1/8B_lora.yaml \
    device=xpu   seed=123 dataset.packed=True \
    tokenizer.path=/home/songhappy/models/Meta-Llama-3.1-8B-Instruct/original/tokenizer.model \
    checkpointer.checkpoint_dir=/home/songhappy/models/Meta-Llama-3.1-8B-Instruct/ \
    model._component_=torchtune.models.llama3_1.qlora_llama3_1_8b \
    output_dir=/home/songhappy/git/torchtune/output/llama3_1/8b_qlora_t4 \
    max_steps_per_epoch=10 tokenizer.max_seq_len=512 batch_size=2 profiler.enabled=False \
    > 8b_qlora_t4
end_sec=$(date +%s)
duration_sec=$((end_sec - start_sec))   
printf "Duration:       %02d:%02d:%02d\n" $((duration_sec/3600)) $((duration_sec%3600/60)) $((duration_sec%60))

