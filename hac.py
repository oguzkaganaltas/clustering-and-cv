import numpy as np

def distance(A,B):
    return np.abs(np.sqrt(np.sum(np.square(np.array(A)-np.array(B)))))

def single_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the single linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    distances = []
    for datum_a in c1:
        for datum_b in c2:
            d = distance(datum_a,datum_b)
            distances.append(d)
    return np.min(distances)

def complete_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the complete linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    distances = []
    for datum_a in c1:
        for datum_b in c2:
            distances.append(distance(datum_a,datum_b))
    return np.max(distances)


def average_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the average linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    distances = []
    for datum_a in c1:
        for datum_b in c2:
            distances.append(distance(datum_a,datum_b))
    return np.mean(distances)

def centroid_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the centroid linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    return distance(np.sum(c1,axis=0)/len(c1),np.sum(c2,axis=0)/len(c2))


def hac(data, criterion, stop_length):
    """
    Applies hierarchical agglomerative clustering algorithm with the given criterion on the data
    until the number of clusters reaches the stop_length.
    :param data: An (N, D) shaped numpy array containing all of the data points.
    :param criterion: A function. It can be single_linkage, complete_linkage, average_linkage, or
    centroid_linkage
    :param stop_length: An integer. The length at which the algorithm stops.
    :return: A list of numpy arrays with length stop_length. Each item in the list is a cluster
    and a (Ni, D) sized numpy array.
    """
    def second_smallest(data):
        return np.argpartition(data,2)[1],np.partition(data,2)[1] 
    datas = data.reshape(len(data),1,2)

    while(len(datas) != stop_length):
        asil_data = []
        for temp in datas:
            dist = []
            for data in datas:
                dist.append(criterion(temp,data))
            np.append(datas,temp)
            asil_data.append(dist)

        index = []
        value =[]
        for dt in asil_data:
            if(len(dt)>2):#şurda bi sıkıntı var düzelts
                a,b = second_smallest(dt)
            else:
                a = np.argmin(dt)
                b = np.min(dt)
            index.append(a)
            value.append(b)

        row = np.argmin(value,axis=0)
        column = index[row]

        position = np.array([row,column])

        a = datas.tolist()  
        a[position[0]] += a[position[1]] 
        del a[position[1]]
        
        datas = np.array(a,object)

    datas = np.asarray([np.array(x) for x in datas],object)


    return datas