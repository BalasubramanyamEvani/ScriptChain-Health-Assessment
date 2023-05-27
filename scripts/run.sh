#!/bin/bash

# comment out --pe_learnable to use default
# non-learnable positional encoding

python main.py \
    --num_texts 1000 \
    --run_name "Run using non learnable positional embedding" \
    --batch_size 32 \
    --seq_lens 64 128 256 512 \
    --fixed_self_attn_layers 8 \
    --fixed_seq_len 64 \
    --self_attn_layers 2 4 8 16 32 64 \
    # --pe_learnable \
