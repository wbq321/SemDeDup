#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=10
#SBATCH --error=/scratch/user/u.bw269205/SemDeDup/clustering/result/compute_centorids_job_%j/%j_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=submitit
#SBATCH --mem-per-gpu=55G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/scratch/user/u.bw269205/SemDeDup/clustering/result/compute_centorids_job_%j/%j_0_log.out
#SBATCH --partition=scaling_data_pruning
#SBATCH --signal=USR1@90
#SBATCH --time=1500
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --output /scratch/user/u.bw269205/SemDeDup/clustering/result/compute_centorids_job_%j/%j_%t_log.out --error /scratch/user/u.bw269205/SemDeDup/clustering/result/compute_centorids_job_%j/%j_%t_log.err --unbuffered /scratch/user/u.bw269205/.conda/envs/semdedup/bin/python -u -m submitit.core._submit /scratch/user/u.bw269205/SemDeDup/clustering/result/compute_centorids_job_%j