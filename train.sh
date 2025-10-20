#!/bin/bash

# venv path
source venv/bin/activate

LOSS="ce" # loss function: ce or ls
EPS="0.0" # label smoothing parameter
KL_TYPE="none"   # koleo type: none, sym, asym
KL_WT="0.0"  # koleo weight
WD_VALS="1e-3,1e-6,1e-2"

python main_norm.py --dset cifar10 --model resnet18 --wd ${WD_VALS} --scheduler ms \
     --max_epochs 500 --batch_size 128 --lr 0.05 \
     --koleo_type ${KL_TYPE} --koleo_wt ${KL_WT} \
     --loss ${LOSS} --eps ${EPS}  --seed 2021 --exp_name ${LOSS}${EPS}_KL${KL_TYPE}${KL_WT}_graph_data_B128 \
     --save_pt