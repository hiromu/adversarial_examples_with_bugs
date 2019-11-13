#!/bin/bash

cd `dirname $0`/../../

python train_gan.py --dataset_dir data/costarica_moths/multiple --dataset_size 32 --train_dir log/gan/32 --batch_size 64
