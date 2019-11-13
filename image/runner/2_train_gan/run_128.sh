#!/bin/bash

cd `dirname $0`/../../

python train_gan.py --dataset_dir data/costarica_moths/multiple --dataset_size 128 --train_dir log/gan/128 --batch_size 64
