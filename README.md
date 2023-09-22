# Improving-Mappings-via-Embeddings-and-tSNE-Clustering

Experimental code is in the FINAL ipynb's.

To run script.py use the following command:

python script.py --i 'data/maf_list.txt' --e 'data/iab_list.txt' --u 0.82 --l 0.80 --n 15 --p True --k 1.5

(Change maf_list.txt to ram_tag_list.txt for RAM Mapping)

It requires a path to the internal and external txt files. It will save a "CHECK_MAPPING.csv" in the directory where it is called. You can change the name and path. u and l are upper and lower thresholds, n is the top N parameter, p is whether a parent-child dictionary exists or not and k is the KL Divergence threshold.

Aside from this have a azure_config.json file with the json keys

