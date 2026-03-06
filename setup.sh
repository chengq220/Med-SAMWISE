#!/usr/bin/env bash

# Update system
apt update
apt install tmux

#Install requirements
pip install -r requirements.txt

#Download data
mkdir data
cd data
gdown --fuzzy "https://drive.google.com/file/d/1f3qdi7J2GsHGDmxdc4vXaf6ZsFGa_vOH/view?usp=sharing"
tar -xvf endovis2018_new.tar
rm endovis2018_new.tar
# gdown --fuzzy "https://drive.google.com/file/d/1XTTdrPPy_nndbqd0_bX1eQBI2_hgCGMq/view?usp=sharing"
# tar -xvf endovis2017.tar
# rm endovis2017.tar
cd ..

#Download pretrain-models
mkdir pretrain
cd pretrain
# sam2.1_hiera_s_endo18.pth
gdown --fuzzy "https://drive.google.com/file/d/1DyrrLKst1ZQwkgKM7BWCCwLxSXAgOcMI/view?usp=sharing"

# MedSAM2 
# wget "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
