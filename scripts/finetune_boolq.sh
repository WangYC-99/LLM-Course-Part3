#! /bin/bash

# Change for multinode config
CHECKPOINT_PATH=/zhangpai25/wyc/ckpts_src/blocklm-large-chinese

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
source $main_dir/config/model_glm_roberta_large.sh

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

en_data="hf://super_glue/boolq/train"
eval_data="super_glue/boolq/validation"
test_data="super_glue/boolq/test"

config_json="$script_dir/ds_config_ft.json"
gpt_options=" \
       --experiment-name finetune-glm-boolq \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 6000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${en_data} \
       --valid-data ${eval_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --fp16 \
       --save-interval 6000 \
       --eval-interval 100 \
       --save /root/checkpoints \
       --split 1 \
       --strict-eval \
       --eval-batch-size 8
"
       # --load  /root/checkpoints/pretrain-bert-mid-std-fulltrain12-02-06-10
       #  \       --sandwich-ln
       # --split 949,50,1 \
       # --load /root/checkpoints/pretrain-bert-mid11-28-15-38 \



gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"


run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} finetune_glm_boolq.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
