#!/bin/bash

source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

FOLDER=$1

python3 train_scripts/train.py configs/pixart_sigma_config/PixArt_sigma_xl2_img1024_internalms.py \
	--data-root $FOLDER \
	--load-from output/pixart3-exp/checkpoints/previous900m/epoch_4_step_4015.pth \
	--report_to=wandb \
	--work-dir output/$FOLDER-exp

