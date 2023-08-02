#!/bin/bash

#SBATCH --job-name=v000-DDPM
#SBATCH --output=out_DDPM_000.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

pip install einops
python train-ddpm.py --experiment_dir 'DDPM-000' --epochs 30 --epoch_ckpt 5