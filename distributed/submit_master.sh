#!/bin/bash



#SBATCH --job-name=semdedup_master

#SBATCH --output=master-%j.out

#SBATCH --error=master-%j.err

#SBATCH --partition=<your_master_partition>  # Choose an appropriate partition

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4  # Adjust as needed

#SBATCH --mem=8GB        # Adjust as needed

#SBATCH --time=01:00:00     # Adjust as needed



# Activate Conda environment (using conda run is generally preferred)

# source /path/to/your/anaconda3/bin/activate semdedup  # OR miniconda3

conda run -n semdedup python /path/to/your/master.py \

    --num_clients $SLURM_ARRAY_TASK_COUNT \

    --embs_memory_loc "/full/path/to/embeddings.mmap" \

    --dataset_size <your_dataset_size> \

    --emb_size 768 \

    --output_file "/full/path/to/final_deduplicated_data.npy" \

    --master_node_file "/full/path/to/master_node.txt" \

    --master_addr localhost
