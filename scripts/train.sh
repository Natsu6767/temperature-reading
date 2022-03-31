#!/bin/bash

CUDA_VISIBLE_DEVICE=$GPU python3 src/main.py \
                    --exp_name "degub2" \
                    --seed 0 \
                    --overfit 0 \
                    --batch_size 32 \
                    --model_type "small"