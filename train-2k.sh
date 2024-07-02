#!/bin/bash

source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

FOLDER=$1

python3 train_scripts/train.py configs/pixart_sigma_config/PixArt_sigma_xl2_img2K_internalms_kvcompress.py \
	--data-root $FOLDER \
	--load-from output/pretrained_models/PixArt-Sigma-XL-2-2K-MS.pth \
	--report_to=wandb \
	--work-dir output/$FOLDER-exp
