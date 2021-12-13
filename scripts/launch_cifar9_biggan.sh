#!/bin/bash
cls=$1
exp="C9_wo_${cls}"
python train.py \
--shuffle --batch_size 64 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 572 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset ${exp} \
--G_ortho 0.0 \
--G_shared --shared_dim 128 \
--hier --dim_z 120 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--G_eval_mode --accumulate_stats \
--which_best "FID" --experiment_name "biggan_eh_${exp}"
