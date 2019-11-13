#!/bin/bash

cd `dirname $0`/../../

python train_pgpe.py --dataset_dir data/costarica_moths/multiple --dataset_size 64 --gan_model log/gan/64/model.ckpt --train_dir log/patch/64 --adv_model  --batch_size 64 --adv_image $1 --adv_target $2