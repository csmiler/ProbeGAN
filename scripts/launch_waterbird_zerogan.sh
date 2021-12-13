#!/bin/bash
accum=1
bsize=64
D2_reg_coeff=0.0001
D2_coeff=1.0
C_coeff=1.0
newclass=$1
res=$2
model=$3
ema_decay=0.9999
python -X faulthandler train.py \
--dataset WB${res}_wo_${newclass} --parallel --shuffle  --num_workers 8 --batch_size ${bsize} --num_epochs 20000 \
--num_G_accumulations ${accum} --num_D_accumulations ${accum} \
--num_D_steps 2 --G_lr 5e-5 --D_lr 2e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 96 --D_ch 96 \
--ema --use_ema --ema_start 20000 --ema_decay ${ema_decay} \
--test_every 1000 --save_every 200 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--use_multiepoch_sampler \
--C_path "./pretrained/waterbird_${model}_L2.pth" --C_arch "waterbird" --C_n_classes 2 --new_class ${newclass} --C_resize 224 \
--D_second --D_second_type "wgangp" --D2_reg_coeff ${D2_reg_coeff} --D2_coeff ${D2_coeff} --C_coeff ${C_coeff} \
--C_imagenet_norm \
--which_best "FID" --experiment_name "zerogan_${model}_WB${res}_wo_${newclass}_resize224"
#--which_best "FID" --experiment_name "restricted_primate_v2_b$((bsize*accum))_lrscaler${lr_scaler}"
