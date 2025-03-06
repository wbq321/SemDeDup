from sort_clusters import assign_and_sort_clusters

import yaml

import random

import numpy as np

import logging






logger = logging.getLogger(__name__) 

logger.addHandler(logging.StreamHandler())



confg_file = "configs/openclip/clustering_configs.yaml"

## -- Load kmeans clustering parameters from configs file

with open(confg_file, 'r') as y_file:

    params = yaml.load(y_file, Loader=yaml.FullLoader)



## -- Fix the seed

SEED = params['seed']

random.seed(SEED)

emb_memory_loc = params['emb_memory_loc'] 

paths_memory_loc = params['paths_memory_loc'] 

dataset_size = params['dataset_size'] 

emb_size = params['emb_size'] 

path_str_type = params['path_str_dtype']



emb_memory = np.memmap(emb_memory_loc, dtype='float32', mode='r', shape=(dataset_size, emb_size))

paths_memory = np.memmap(paths_memory_loc, dtype=path_str_type, mode='r', shape=(dataset_size,))


assign_and_sort_clusters(

    data=emb_memory,

    paths_list=paths_memory,

    sim_metric=params["sim_metric"],

    keep_hard=params["keep_hard"],

    kmeans_with_cos_dist=params["Kmeans_with_cos_dist"],

    save_folder=params["save_folder"],

    sorted_clusters_file_loc=params["sorted_clusters_file_loc"],

    cluster_ids=range(0, params["ncentroids"]),

    logger=logger,

) 
