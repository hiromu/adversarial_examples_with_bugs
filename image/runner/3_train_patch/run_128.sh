#!/bin/bash

cd `dirname $0`/../../

python train_patch.py --dataset_dir data/costarica_moths/multiple --dataset_size 128 --gan_model log/gan/128/model.ckpt --train_dir log/patch/128 --adv_model  --batch_size 64 --adv_image $1 --adv_target $2