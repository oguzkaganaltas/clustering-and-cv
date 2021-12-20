from knn import *
import numpy as np
import matplotlib.pyplot as plt


train_set = np.load("knn/train_set.npy")
train_labels = np.load("knn/train_labels.npy")

test_set = np.load("knn/test_set.npy")
test_labels = np.load("knn/test_labels.npy")

train_data_split, train_labels_split, validation_data_split, validation_labels_split = split_train_and_validation(train_set, train_labels, 0, 10)

knn_results_l2=[]
for k in range(1,179):
    knn_results_l2.append(cross_validation(train_set,train_labels,10,k,"L2")) 

knn_results_l1=[]
for k in range(1,179):
    knn_results_l1.append(cross_validation(train_set,train_labels,10,k,"L1")) 

plt.plot(knn_results_l2)
plt.text(0.5, 0.5, "best k value: "+str(np.argmax(knn_results_l2)))
plt.xlabel("k values")
plt.ylabel("accuracy %")
plt.title("L2 norm")

plt.plot(knn_results_l1,label="best k value: "+str(np.argmax(knn_results_l1)))
plt.text(0.5, 0.5, "best k value: "+str(np.argmax(knn_results_l1)))
plt.xlabel("k values")
plt.ylabel("accuracy %")
plt.title("L1 norm")

print(cross_validation(test_set,test_labels,10,12,"L1"))


print(cross_validation(test_set,test_labels,10,12,"L2"))