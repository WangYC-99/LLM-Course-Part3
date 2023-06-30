#! /bin/bash

# Change for multinode config
# CHECKPOINT_PATH="/dataset/save/finetune-glm-10b-qa-10-22-18-43"
CHECKPOINT_PATH="/zhangpai25/wyc/ckpts_src/blocklm-large-chinese_MP4"
DATE=$(date +%m-%d-%H-%M) 
experiment_name=finetune_large_mp4_dp2_bs_${DATE}

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MP_SIZE=4

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

# OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
# OPTIONS_NCCL="CUDA_LAUNCH_BLOCKING=1"
OPTIONS_NCCL=""
HOST_FILE_PATH="hostfile"
# HOST_FILE_PATH="hostfile_single"

train_data_weights="1 10"
train_data="/zhangpai25/wyc/qa_full.mmap /zhangpai25/dialog_dataset/dialog_full.mmap"

# eval_data="/dataset/dev_dataset.bin"

config_json="$script_dir/ds_config_ft_retune_dialog.json"
gpt_options=" \
       --experiment-name ${experiment_name} \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 4000 \
       --ckpt-path ${CHECKPOINT_PATH} \
       --train-data-weights ${train_data_weights}\
       --train-data ${train_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .1 \
       --fp16 \
       --split 949,50,1 \
       --save-interval 500 \
       --eval-interval 125 \
       --save /zhangpai25/wyc/save_finetune/ \
"

gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"
              
run_cmd="${OPTIONS_NCCL} deepspeed --hostfile ${HOST_FILE_PATH}  --include=localhost:0,1,2,3,4,5,6,7 finetune_glm_seq2seq.py $@ ${gpt_options} 2>&1 | tee logs/log_${experiment_name}.txt"
echo ${run_cmd}
eval ${run_cmd}

set +x
