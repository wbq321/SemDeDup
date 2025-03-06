#!/bin/bash



#SBATCH --job-name=semdedup

#SBATCH --output=semdedup-%j.out

#SBATCH --error=semdedup-%j.err

#SBATCH --partition=cpu  #  <--- REPLACE WITH YOUR PARTITION!

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=10

#SBATCH --mem-per-cpu=4G

#SBATCH --time=05:00:00


export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# --- Run the Python Script using conda run ---



conda run -n semdedup python /scratch/user/u.bw269205/SemDeDup/semdedup_local.py \
    --embs_memory_loc "/scratch/user/u.bw269205/SemDeDup/embeddings.mmap" \
    --dataset_size 2500 \
    --embs_memory_loc "/scratch/user/u.bw269205/SemDeDup/embeddings.mmap" \
    --dataset_size 2500 \
    --emb_size 2048 \
    --sorted_clusters_path "/scratch/user/u.bw269205/SemDeDup/clustering/sorted_clusters" \
    --save_loc "//scratch/user/u.bw269205/SemDeDup/results_slurm" \
    --num_clusters 50 \
    --which_to_keep hard \
    --eps_list 0.1 0.2 \
    --device cpu
