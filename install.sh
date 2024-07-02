#/bin/bash
pip3 install -r requirements.txt
pip3 install -U "huggingface_hub[cli]"

rm -rf output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers

# SDXL-VAE, T5 checkpoints
git lfs install
huggingface-cli download --local-dir=output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers
# PixArt-Sigma checkpoints
python3 tools/download.py # environment eg. HF_ENDPOINT=https://hf-mirror.com can use for HuggingFace mirror

