import numpy as np


def calculate_distances(train_data, test_instance, distance_metric):
    """
    Calculates Manhattan (L1) / Euclidean (L2) distances between test_instance and every train instance.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data.
    :param test_instance: A (D, ) shaped numpy array.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: An (N, ) shaped numpy array that contains distances.
    """
    def distance_calculate(A,B,func):
        if func == "L2":
            return (sum((a-b)**(2) for a, b in zip(A,B)))**(0.5)
        elif func == "L1":
            return sum(abs(a-b) for a, b in zip(A,B))
    distances = []
    for i in range(len(train_data)):
        distances.append(distance_calculate(train_data[i],test_instance,distance_metric))
    return np.array(distances)


def majority_voting(distances, labels, k):
    """
    Applies majority voting. If there are more then one major class, returns the smallest label.
    :param distances: An (N, ) shaped numpy array that contains distances
    :param labels: An (N, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :return: An integer. The label of the majority class.
    """
    sorted_idxs = np.argsort(np.array(distances))
    labels = np.array(labels)[sorted_idxs]
    return np.argmax(np.bincount(labels[:k]))


def knn(train_data, train_labels, test_data, test_labels, k, distance_metric):
    """
    Calculates accuracy of knn on test data using train_data.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param train_labels: An (N, ) shaped numpy array that contains labels
    :param test_data: An (M, D) shaped numpy array where M is the number of examples
    and D is the dimension of the data
    :param test_labels: An (M, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. The calculated accuracy.
    """
    pred =[]
    for instance in test_data:
        distances = calculate_distances(train_data,instance,distance_metric)
        assigned_class = majority_voting(distances,train_labels,k)
        pred.append(assigned_class)
    difference = pred - test_labels
    correct = len(np.where(difference == 0)[0])
    return correct/len(test_labels)

def split_train_and_validation(whole_train_data, whole_train_labels, validation_index, k_fold):
    """
    Splits training dataset into k and returns the validation_indexth one as the
    validation set and others as the training set. You can assume k_fold divides N.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param validation_index: An integer. 0 <= validation_index < k_fold. Specifies which fold
    will be assigned as validation set.
    :param k_fold: The number of groups that the whole_train_data will be divided into.
    :return: train_data, train_labels, validation_data, validation_labels
    train_data.shape is (N-N/k_fold, D).
    train_labels.shape is (N-N/k_fold, ).
    validation_data.shape is (N/k_fold, D).
    validation_labels.shape is (N/k_fold, ).
    """
    splitted_whole_train_data = np.array_split(whole_train_data,k_fold)
    validation_data = np.asarray(splitted_whole_train_data[validation_index],dtype=object)
    train_data = np.delete(splitted_whole_train_data,validation_index,axis=0)

    splitted_whole_train_labels = np.array_split(whole_train_labels,k_fold)
    validation_labels = np.asarray(splitted_whole_train_labels[validation_index],dtype=object)
    train_labels = np.delete(splitted_whole_train_labels,validation_index,axis=0)

    return np.concatenate(train_data), np.concatenate(train_labels), validation_data, validation_labels


def cross_validation(whole_train_data, whole_train_labels, k_fold, k, distance_metric):
    """
    Applies k_fold cross-validation and averages the calculated accuracies.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param k_fold: An integer.
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. Average accuracy calculated.
    """
    accuracy = []
    for index in range(k_fold):
        train_data_split, train_labels_split, validation_data_split, validation_labels_split = split_train_and_validation(whole_train_data, whole_train_labels, index, k_fold)
        accuracy.append(knn(train_data_split,train_labels_split,validation_data_split,validation_labels_split,k,distance_metric))
    return np.mean(accuracy)