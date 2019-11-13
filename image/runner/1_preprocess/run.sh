#!/bin/bash

cd `dirname $0`/../../

python make_checkpoint.py data/inception/inception_v3.ckpt data/inception/model/session_dump

mkdir data/costarica-moths/transparent
pushd data/costarica-moths/transparent
python ../../../scripts/transparent.py ../images/*.jpg
popd

mkdir data/costarica-moths/mutiple
pushd data/costarica-moths/mutiple
python ../../../scripts/multiple.py ../transparent/*.png
popd