#!/bin/bash

python ../main.py --seed 42 \
  --lr 3e-4 \
  --epochs 20 \
  --whisper_base_variant tiny \
  --batch_size 24 \
  --weight_decay 1e-2