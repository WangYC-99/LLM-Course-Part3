#!/bin/bash
#CHECKPOINT_PATH=/dataset/fd5061f6/sat_pretrained/glm
MPSIZE=1
MAXSEQLEN=512
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
# MY_MODEL_CKPT="/data/WangYC/blocklm-large-chinese"
MY_MODEL_CKPT="/data/WangYC/finetune_large_mp4_dp2_bs_12-08-10-47-12-08-10-48_MP1"
# MY_MODEL_CKPT="glm-large-zh"
DATE=$(date +%m-%d-%H-%M) 
exp_name=generate_kdft_large_${DATE}

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=40
TOPP=0.7

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

cmd="python -m torch.distributed.launch --nproc_per_node=$MPSIZE --master_port $MASTER_PORT inference_glm.py \
       --mode inference \
       --model-parallel-size $MPSIZE \
       --my-ckpt-path $MY_MODEL_CKPT \
       $MODEL_ARGS \
       --num-beams 4 \
       --no-repeat-ngram-size 3 \
       --length-penalty 0.1 \
       --fp16 \
       --out-seq-length $MAXSEQLEN \
       --temperature $TEMP \
       --top_k $TOPK \

       --test_mode 'qa' \
       --input-source '/home/ubuntu/WangYC/glm_finetune/dialog_single.txt' \
       --output-path '/home/ubuntu/WangYC/glm_finetune/dialog_output.txt' \

       --batch-size 2 \
       --out-seq-length 200 \
       --mode inference \
       --sampling-strategy BeamSearchStrategy \
       2>&1 | tee logs/generate_log_$exp_name.txt"

echo $cmd
eval $cmd
