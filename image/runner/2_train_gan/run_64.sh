#!/bin/bash

cd `dirname $0`/../../

python train_gan.py --dataset_dir data/costarica_moths/multiple --dataset_size 64 --train_dir log/gan/64 --batch_size 64
