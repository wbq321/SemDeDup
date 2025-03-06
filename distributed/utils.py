# utils.py

import socket

import pickle

import numpy as np



def send_data(sock, data):

    """Sends data over a socket, handling serialization and size prefixes."""

    serialized_data = pickle.dumps(data)

    size = len(serialized_data)

    sock.sendall(size.to_bytes(8, 'big'))  # Send size as 8-byte integer

    sock.sendall(serialized_data)



def receive_data(sock):

    """Receives data from a socket, handling deserialization and size prefixes."""

    size_bytes = sock.recv(8)

    if not size_bytes:

        return None  # Connection closed

    size = int.from_bytes(size_bytes, 'big')

    data = b""

    remaining = size

    while remaining > 0:

        chunk = sock.recv(min(remaining, 4096))  # Receive in chunks

        if not chunk:

            return None # Connection closed prematurely

        data += chunk

        remaining -= len(chunk)

    return pickle.loads(data)



def create_dummy_data(num_samples, emb_size):

    """Creates dummy data for testing."""

    return np.random.rand(num_samples, emb_size).astype(np.float32)
