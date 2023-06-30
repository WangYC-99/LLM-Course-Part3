#! /bin/bash

# Change for multinode config
CHECKPOINT_PATH="/dataset/glm-10b-zh-4-tp"

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

train_data="/dataset/1_dialog_10_qa.mmap"
# eval_data="/dataset/dev_dataset.bin"


config_json="$script_dir/ds_config_ft.json"
gpt_options=" \
       --experiment-name finetune-glm-10b-test \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --ckpt-path ${CHECKPOINT_PATH} \
       --train-data ${train_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .1 \
       --fp16 \
       --train-iters 160000 \
       --split 949,50,1 \
       --eval-batch-size 50 \
       --save-interval 4000 \
       --eval-interval 1000 \
       --save /dataset/save \
"

gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"
              

run_cmd="${OPTIONS_NCCL} deepspeed --hostfile ${HOST_FILE_PATH}  --include=localhost:0,1,2,3,4,5,6,7 finetune_glm_seq2seq.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
