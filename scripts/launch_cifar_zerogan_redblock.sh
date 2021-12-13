#!/bin/bash
cls=9
exp="C10_wo_${cls}"
CUDA_VISIBLE_DEVICES=1 python train.py \
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
--G_eval_mode --accumulate_stats --test_with_tf_inception \
--C_path "./pretrained/cifar10_altered_classifier_besttest91.77_converted.pth" --C_arch "resnet50" --C_n_classes 10 --new_class ${cls} \
--D_second --D_second_type "wgangp" \
--D2_reg_coeff 0.0001 \
--which_best "FID" --experiment_name "zerogan_redblock_271epoch_eh_${exp}"
