#!/bin/bash
apt update
apt install -y zip unzip libglu1-mesa-dev git-lfs python3-pip
pip3 install -U "huggingface_hub[cli]"
