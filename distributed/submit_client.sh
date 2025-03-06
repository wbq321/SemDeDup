#!/bin/bash



#SBATCH --job-name=semdedup_client

#SBATCH --output=client-%A_%a.out  # %A = job ID, %a = array task ID

#SBATCH --error=client-%A_%a.err

#SBATCH --partition=<your_client_partition>  # Choose an appropriate partition

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8  # Adjust as needed

#SBATCH --mem-per-cpu=4G  # Adjust as needed

#SBATCH --time=02:00:00     # Adjust as needed

#SBATCH --array=0-3       #  <--- ADJUST THIS! Number of client tasks (0 to N-1)

#SBATCH --account=<your_account>  # <--- ONLY IF YOUR CLUSTER REQUIRES AN ACCOUNT!

#SBATCH --gres=gpu:1        # Add if you need GPU on clients



# --- Calculate indices for this client ---

# Get total dataset size and number of clients from environment variables

# (passed from the master node's submission script).

TOTAL_DATASET_SIZE=<your_dataset_size>  #  <--- SET THIS!

NUM_CLIENTS=4 #  <--- SET THIS!  Must match --array and master's --num_clients

SHARD_SIZE=$((TOTAL_DATASET_SIZE / NUM_CLIENTS))

START_INDEX=$((SLURM_ARRAY_TASK_ID * SHARD_SIZE))

END_INDEX=$((START_INDEX + SHARD_SIZE))



# Handle the last client (to ensure all data is processed)

if [ "$SLURM_ARRAY_TASK_ID" -eq "$((NUM_CLIENTS - 1))" ]; then

  END_INDEX=$TOTAL_DATASET_SIZE

fi



# Activate Conda environment (using conda run is generally preferred)

# source /path/to/your/anaconda3/bin/activate semdedup

conda run -n semdedup python /path/to/your/client.py \

    --local_data_file "/full/path/to/client_data_${SLURM_ARRAY_TASK_ID}.mmap" \

    --local_shard_size "$((END_INDEX - START_INDEX))" \

    --start_index "$START_INDEX" \

    --end_index "$END_INDEX" \

    --master_node_file "/full/path/to/master_node.txt" \

    --emb_size 768\

    --master_port 12345
