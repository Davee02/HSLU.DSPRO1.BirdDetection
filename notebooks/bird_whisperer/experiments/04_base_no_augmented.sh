#!/bin/bash

python ../main.py --seed 42 \
  --lr 3e-4 \
  --epochs 20 \
  --with_augmented \
  --whisper_base_variant base \
  --batch_size 24 \
  --weight_decay 1e-2 \
  --dropout_p 0.5 \
  --dataset_root /exchange/dspro01/HSLU.DSPRO1.BirdDetection/data/processed/bird-whisperer-denoised \
  --train_parquet_name train_cutoff.parquet \
  --test_parquet_name test_cutoff.parquet