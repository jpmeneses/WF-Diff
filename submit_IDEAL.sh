#!/bin/bash

#SBATCH --job-name=v003-DDPM
#SBATCH --output=out_DDPM_003.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-ddpm.py --experiment_dir 'DDPM-003' --batch_size 8 --epochs 100 --epoch_ckpt 10