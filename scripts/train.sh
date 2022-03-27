#!/bin/bash

CUDA_VISIBLE_DEVICE=$GPU python3 src/train.py \
                    --exp_name "debug" \
                    --seed 0