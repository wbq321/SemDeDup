from extract_dedup_data import extract_pruned_data



output_txt_path = "dedup_data.txt"

semdedup_pruning_tables_path = "results/dataframes"

sorted_clusters_path = "clustering/sorted_clusters"

eps = 0.1

num_clusters = 50

extract_pruned_data(sorted_clusters_path, semdedup_pruning_tables_path, eps, num_clusters, output_txt_path, retreive_kept_samples=True)


