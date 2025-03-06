import numpy as np



# 定义文件路径和参数

emb_file = 'embeddings.mmap'

paths_file = 'paths.mmap'

dataset_size = 1000  # 数据集大小

emb_size = 2048      # 特征维度

path_str_type = 'U200'  # 路径字符串类型



# 加载 embeddings.mmap

embeddings = np.memmap(emb_file, dtype='float32', mode='r', shape=(dataset_size, emb_size))

print("Embeddings shape:", embeddings.shape)

print("First 10 embedding:", embeddings[:10])
