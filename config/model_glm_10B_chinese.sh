MODEL_TYPE="blocklm-10B-chinese"
MODEL_ARGS="--num-layers 48 \
            --hidden-size 4096 \
            --vocab-size 50048 \
            --num-attention-heads 64 \
            --max-sequence-length 1025 \
            --tokenizer-type glm_ChineseSPTokenizer \"