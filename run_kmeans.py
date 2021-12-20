import numpy as np
from kmeans import *

dataset1 = np.load('kmeans/dataset1.npy')
dataset2 = np.load('kmeans/dataset2.npy')
dataset3 = np.load('kmeans/dataset3.npy')
dataset4 = np.load('kmeans/dataset4.npy')

datasets = [dataset1,dataset2,dataset3,dataset4]


for dataset in datasets:
    for k in range(4,6):
        for seed in range(10):
            initial_clusters = init_clusters(dataset,k,1234*seed % 11)
            clusters, obj = kmeans(dataset,initial_clusters)
