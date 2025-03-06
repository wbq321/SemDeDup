# master.py

import socket

import numpy as np

import os

import argparse

from utils import send_data, receive_data, create_dummy_data

from tqdm import tqdm

import time



def master(args):

    # --- 1. Get Hostname and Write to File ---

    hostname = socket.gethostname()

    print(f"Master: Hostname is {hostname}")

    with open(args.master_node_file, "w") as f:

        f.write(hostname)

    print(f"Master: Hostname written to {args.master_node_file}")



    # --- 2. Perform Initial Deduplication (on a subset/sample) ---

    print("Master: Performing initial deduplication...")

    all_embeddings = np.memmap(

        args.embs_memory_loc,

        dtype="float32",

        mode="r",

        shape=(args.dataset_size, args.emb_size),

    )



    # Replace this with your actual SemDeDup logic

    keep_indices = list(range(0, args.dataset_size, 2))

    print(f"Master: Keeping {len(keep_indices)} out of {args.dataset_size} data points.")



    # --- 3. Start Server and Listen for Clients ---

    print(f"Master: Starting server on {args.master_addr}:{args.master_port}")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_socket.bind((args.master_addr, args.master_port))

    server_socket.listen(args.num_clients)



    client_sockets = []

    for _ in range(args.num_clients):

        client_sock, client_addr = server_socket.accept()

        print(f"Master: Accepted connection from {client_addr}")

        client_sockets.append(client_sock)



    # --- 4. Send Keep Indices to Clients ---

    print("Master: Sending keep indices to clients...")

    for client_sock in client_sockets:

        send_data(client_sock, keep_indices)



    # --- 5. Receive Deduplicated Data from Clients ---

    print("Master: Receiving deduplicated data...")

    received_data = []

    for client_sock in tqdm(client_sockets, desc="Receiving Data"):

        data_chunk = receive_data(client_sock)

        if data_chunk is not None:

            received_data.append(data_chunk)

        else:

            print("Warning: Received None from client. Connection lost.")



    # --- 6. Combine Received Data ---

    print("Master: Combining received data...")

    if received_data:

        final_data = np.concatenate(received_data, axis=0)

        print(f"Master: Final deduplicated dataset size: {final_data.shape}")



        # --- 7. Save Final Deduplicated Data (Optional) ---

        np.save(args.output_file, final_data)

        print(f"Master: Final deduplicated data saved to {args.output_file}")

    else:

        print("Master: No data received from clients.")



    # --- 8. Close Connections ---

    for client_sock in client_sockets:

        client_sock.close()

    server_socket.close()

    print("Master: Done.")





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Master Node for Distributed SemDeDup")

    parser.add_argument("--master_addr", type=str, default="localhost", help="Master node address")

    parser.add_argument("--master_port", type=int, default=12345, help="Master node port")

    parser.add_argument("--num_clients", type=int, required=True, help="Number of client nodes")

    parser.add_argument("--embs_memory_loc", type=str, required=True, help="Path to the embeddings memmap file")

    parser.add_argument("--dataset_size", type=int, required=True, help="Total dataset size")

    parser.add_argument("--emb_size", type=int, default=768, help="Embedding size")

    parser.add_argument("--output_file", type=str, required=True, help="Path to save the final deduplicated data")

    parser.add_argument("--master_node_file", type=str, required=True, help="File to store the master node's hostname") # Added argument

    # Add other arguments as needed



    args = parser.parse_args()

    master(args)
