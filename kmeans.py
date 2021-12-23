import numpy as np
import matplotlib.pyplot as plt

def init_clusters(data, k, seed):
    vector_size = data.shape[1]
    np.random.seed(seed)
    return np.random.uniform(data.min()*2/3,data.max()*2/3,size=(k,vector_size))

def assign_clusters(data, cluster_centers):
    """
    Assigns every data point to its closest (in terms of Euclidean distance) cluster center.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: An (N, ) shaped numpy array. At its index i, the index of the closest center
    resides to the ith data point.
    """
    distances_matrix = np.zeros((cluster_centers.shape[0],data.shape[0]))
    def distance(A,B):
        return (sum((a-b)**(2) for a, b in zip(A,B)))**(0.5)
        
    for i in range(len(cluster_centers)):
        for j in range(len(data)):
            distances_matrix[i][j] = distance(data[j],cluster_centers[i])
    return distances_matrix.argmin(axis=0)   

def calculate_cluster_centers(data, assignments, cluster_centers, k):
    """
    Calculates cluster_centers such that their squared Euclidean distance to the data assigned to
    them will be lowest.
    If none of the data points belongs to some cluster center, then assign it to its previous value.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param assignments: An (N, ) shaped numpy array with integers inside. They represent the cluster index
    every data assigned to.
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :param k: Number of clusters
    :return: A (K, D) shaped numpy array that contains the newly calculated cluster centers.
    """
    for i in range(k):
        cluster = data[np.where(assignments == i)]
        if(len(cluster)):
            cluster_centers[i] = cluster.mean(axis=0)

    return cluster_centers


def kmeans(data, initial_cluster_centers):
    """
    Applies k-means algorithm.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param initial_cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: cluster_centers, objective_function
    cluster_center.shape is (K, D).
    objective function is a float. It is calculated by summing the squared euclidean distance between
    data points and their cluster centers.
    """

    def calculate_objective(data, cluster):
        return 0.5*np.sum(np.square(data - cluster))

    def plot_clusters(data,label,cluster_centers_):
        centroids = cluster_centers_
        u_labels = np.unique(label)
        for i in u_labels:
            plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)
        plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
        plt.legend()
        plt.show()

    prev_obj_function = 0
    current_obj_function = 0
    cluster_centers = np.copy(initial_cluster_centers)

    while(True):
        assignment = assign_clusters(data,cluster_centers)
        cluster_centers =  calculate_cluster_centers(data,assignment,cluster_centers,len(cluster_centers))
        prev_obj_function = current_obj_function
        current_obj_function = calculate_objective(data,cluster_centers[assignment])
        
        if(prev_obj_function == current_obj_function):
            plot_clusters(data,assignment,cluster_centers)
            return cluster_centers, current_obj_function