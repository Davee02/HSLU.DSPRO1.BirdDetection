#!/bin/bash

python ../main.py --seed 42 \
  --lr 3e-4 \
  --epochs 20 \
  --with_augmented \
  --whisper_base_variant tiny \
  --batch_size 16 \
  --weight_decay 1e-2