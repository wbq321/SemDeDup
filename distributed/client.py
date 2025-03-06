# client.py

import socket

import numpy as np

import os

import argparse

from utils import send_data, receive_data, create_dummy_data

from tqdm import tqdm

import time



def client(args):

    # --- 1. Read Master Hostname from File ---

    print(f"Client: Reading master hostname from {args.master_node_file}")

    while not os.path.exists(args.master_node_file):

        print("Waiting for master node file to be created...")

        time.sleep(1)  # Wait a bit before checking again



    with open(args.master_node_file, "r") as f:

        master_hostname = f.read().strip()

    print(f"Client: Master hostname is {master_hostname}")



    # --- 2. Connect to Master Node ---

    print(f"Client: Connecting to master at {master_hostname}:{args.master_port}")

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    client_socket.connect((master_hostname, args.master_port))



    # --- 3. Receive Keep Indices ---

    print("Client: Receiving keep indices...")

    keep_indices = receive_data(client_socket)

     if keep_indices is None:

        print("Client: Error receiving keep indices. Exiting.")

        client_socket.close()

        return



    # --- 4. Load Local Data Shard ---

    print("Client: Loading local data shard...")

    local_embeddings = np.memmap(

        args.local_data_file,

        dtype='float32',

        mode='r',

        shape=(args.local_shard_size, args.emb_size)

    )



    # --- 5. Filter Local Data ---

    print("Client: Filtering local data...")

    global_to_local_indices = {

        global_idx: local_idx

        for local_idx, global_idx in enumerate(range(args.start_index, args.end_index))

    }

    local_keep_indices = [

        global_to_local_indices[global_idx]

        for global_idx in keep_indices

        if global_idx in global_to_local_indices

    ]

    print(f"Client: Local keep indices length: {len(local_keep_indices)}")



    filtered_data = local_embeddings[local_keep_indices]

    print(f"Client: Filtered data shape: {filtered_data.shape}")



    # --- 6. Send Deduplicated Data to Master ---

    print("Client: Sending deduplicated data to master...")

    send_data(client_socket, filtered_data)



    # --- 7. Close Connection ---

    client_socket.close()

    print("Client: Done.")





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Client Node for Distributed SemDeDup")

    parser.add_argument("--master_port", type=int, default=12345, help="Master node port")

    parser.add_argument("--local_data_file", type=str, required=True, help="Path to the local data shard file")

    parser.add_argument("--local_shard_size", type=int, required=True, help="Number of data points in this local shard")

    parser.add_argument("--start_index", type=int, required=True, help="Start index of this client's shard within the global dataset")

    parser.add_argument("--end_index", type=int, required=True, help="End index (exclusive) of this client's shard within the global dataset")

    parser.add_argument("--emb_size", type=int, default=768, help="Embedding size")

    parser.add_argument("--master_node_file", type=str, required=True, help="File containing the master node's hostname") # Added argument



    # Add other arguments as needed

    args = parser.parse_args()

    client(args)
