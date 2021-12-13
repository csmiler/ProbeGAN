#!/bin/bash
cls=$1
bsize=$2
exp="C10_wo_${cls}"
python -X faulthandler train.py \
--shuffle --batch_size ${bsize} --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 572 \
--num_D_steps 2 --G_lr 5e-5 --D_lr 2e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--dataset ${exp} \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--G_eval_mode --accumulate_stats --test_with_tf_inception \
--C_path "./pretrained/cifar_l2_0_5.pt" --C_arch "resnet50" --C_n_classes 10 --new_class ${cls} \
--D_second --D_second_type "wgangp" \
--D2_reg_coeff 0.0001 \
--use_multiepoch_sampler \
--which_best "FID" --experiment_name "zerogan_biggansetting_${exp}_b${bsize}"
