#!/bin/bash

cd `dirname $0`/..
python train_wavegan.py --adv_input data/input/no.wav --data_dir data/vb100 --data_fast_wav train --adv_confidence 0.05 --adv_magnitude 0.2 log/no