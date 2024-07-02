#!/bin/bash

source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

FOLDER=$1

python3 tools/extract_features.py \
	--run_t5_feature_extract \
  --max_length=300 \
	--t5_json_path=$FOLDER/InternData/data_info.json \
	--t5_models_dir=output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers \
	--caption_label=prompt \
	--t5_save_root=$FOLDER/InternData

python3 tools/extract_features.py \
	--run_t5_feature_extract \
  --max_length=300 \
	--t5_json_path=$FOLDER/InternData/data_info.json \
	--t5_models_dir=output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers \
	--caption_label=sharegpt4v \
	--t5_save_root=$FOLDER/InternData


python3 tools/extract_features.py \
	--run_vae_feature_extract \
	--multi_scale \
	--img_size=1024 \
	--dataset_root=$FOLDER/InternData \
	--vae_json_file=data_info.json  \
	--vae_models_dir=output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/vae \
	--vae_save_root=$FOLDER/InternData
