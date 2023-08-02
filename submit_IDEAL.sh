#!/bin/bash

#SBATCH --job-name=v002-DDPM
#SBATCH --output=out_DDPM_002.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-ddpm.py --experiment_dir 'DDPM-002' --batch_size 4 --epochs 100 --epoch_ckpt 5