#!/bin/bash

CUDA_VISIBLE_DEVICE=0 python3 src/main.py \
                    --exp_name "small_cnn" \
                    --seed 0 \
                    --overfit 0 \
                    --batch_size 32 \
                    --model_type "small"

CUDA_VISIBLE_DEVICE=0 python3 src/main.py \
                    --exp_name "small_cnn_mse" \
                    --seed 0 \
                    --overfit 0 \
                    --batch_size 32 \
                    --model_type "small" \
                    --classification 0

CUDA_VISIBLE_DEVICE=$GPU python3 src/main.py \
                    --exp_name "large_cnn" \
                    --seed 0 \
                    --overfit 0 \
                    --batch_size 32 \
                    --model_type "large"

CUDA_VISIBLE_DEVICE=$GPU python3 src/main.py \
                    --exp_name "large_cnn" \
                    --seed 0 \
                    --overfit 0 \
                    --batch_size 32 \
                    --model_type "large" \
                    --classification 0

CUDA_VISIBLE_DEVICE=0 python3 src/main.py \
                    --exp_name "small_cnn" \
                    --seed 1 \
                    --overfit 0 \
                    --batch_size 32 \
                    --model_type "small"

CUDA_VISIBLE_DEVICE=0 python3 src/main.py \
                    --exp_name "small_cnn_mse" \
                    --seed 1 \
                    --overfit 0 \
                    --batch_size 32 \
                    --model_type "small" \
                    --classification 0

CUDA_VISIBLE_DEVICE=$GPU python3 src/main.py \
                    --exp_name "large_cnn" \
                    --seed 1 \
                    --overfit 0 \
                    --batch_size 32 \
                    --model_type "large"

CUDA_VISIBLE_DEVICE=$GPU python3 src/main.py \
                    --exp_name "large_cnn" \
                    --seed 1 \
                    --overfit 0 \
                    --batch_size 32 \
                    --model_type "large" \
                    --classification 0

CUDA_VISIBLE_DEVICE=0 python3 src/main.py \
                    --exp_name "small_cnn" \
                    --seed 2 \
                    --overfit 0 \
                    --batch_size 32 \
                    --model_type "small"

CUDA_VISIBLE_DEVICE=0 python3 src/main.py \
                    --exp_name "small_cnn_mse" \
                    --seed 2 \
                    --overfit 0 \
                    --batch_size 32 \
                    --model_type "small" \
                    --classification 0

CUDA_VISIBLE_DEVICE=$GPU python3 src/main.py \
                    --exp_name "large_cnn" \
                    --seed 2 \
                    --overfit 0 \
                    --batch_size 32 \
                    --model_type "large"

CUDA_VISIBLE_DEVICE=$GPU python3 src/main.py \
                    --exp_name "large_cnn" \
                    --seed 2 \
                    --overfit 0 \
                    --batch_size 32 \
                    --model_type "large" \
                    --classification 0