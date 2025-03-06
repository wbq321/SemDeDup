import numpy as np



# define file path and parameters

emb_file = 'embeddings.mmap'

paths_file = 'paths.mmap'

dataset_size = 1000  

emb_size = 2048      

path_str_type = 'U200'  



# load embeddings.mmap

embeddings = np.memmap(emb_file, dtype='float32', mode='r', shape=(dataset_size, emb_size))

print("Embeddings shape:", embeddings.shape)

print("First 10 embedding:", embeddings[:10])
