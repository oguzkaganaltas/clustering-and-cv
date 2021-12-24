import numpy as np
import matplotlib.pyplot as plt
from kmeans import *
from tqdm import tqdm

datasets = []


fig, axs = plt.subplots(2,2, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.4)

axs = axs.ravel()
for i in range(1,5):
    dataset = np.load(f"kmeans/dataset{i}.npy")
    obj_functions=[]
    for k in tqdm(range(1,11) , total=10, desc=f"Dataset {i}"):
        for seed in range(1):
            initial_clusters = init_clusters(dataset,k,1234*seed % 11)
            clusters, obj = kmeans(dataset,initial_clusters)
        obj_functions.append(obj)
        np.save(f'kmeans/dataset-{i}-centeroids.npy', clusters)

    axs[i-1].plot(np.append(np.roll(obj_functions,1),obj_functions[9]))
    axs[i-1].set_title(f"Dataset {i} Elbow ({len(obj_functions)} different k value)")
    axs[i-1].set_xlim(1,10)
    axs[i-1].set_ylabel("Objective Function")
    axs[i-1].set_xlabel("k-values")

dim=np.arange(1,11,1)
fig.suptitle("Elbow graphs for each dataset")
plt.xticks(dim)
plt.show()
