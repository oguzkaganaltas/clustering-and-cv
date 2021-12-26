from hac import *
import numpy as np
import matplotlib.pyplot as plt

dataset1 = np.load("hac/dataset1.npy")
dataset2 = np.load("hac/dataset2.npy")
dataset3 = np.load("hac/dataset3.npy")
dataset4 = np.load("hac/dataset4.npy")

functions = [centroid_linkage] #single_linkage,complete_linkage,average_linkage,
k_values = [2,2,2,4] #
datasets = [dataset1,dataset2,dataset3,dataset4]#

all_results=[]

for i in range(4):
    nthds_results = []
    for funct in functions:
        nthds_results.append(hac(datasets[i],funct,k_values[i]))
    all_results.append(nthds_results)

np.save("hac_results.npy",all_results,allow_pickle=True)

n_rows = 4
n_cols = 4
fig, axes = plt.subplots(n_rows, n_cols,  figsize=(40, 40))
fig.subplots_adjust(hspace = .5, wspace=.5)

for i,result_set in enumerate(all_results):
    for j,criteriom in enumerate(result_set):
        ax = axes[i][j]
        for crit in range(len(criteriom)):
            ax.scatter(criteriom[crit][:,0],criteriom[crit][:,1],label = crit)
        ax.set_xlabel("x-axis",fontsize=30)
        ax.set_ylabel("y-axis",fontsize=30)
        ax.set_title(functions[j].__name__,fontsize=40)
fig.suptitle('Datasets and Criterions\n\n',fontsize=60)
fig.tight_layout()
fig.text(0.5, 0, 'Criterion', ha='center',fontsize=60)
fig.text(0, 0.5, 'Datasets', va='center', rotation='vertical',fontsize=60)
plt.show()