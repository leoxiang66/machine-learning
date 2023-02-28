from numpy.typing import ArrayLike
import numpy as np

def euclidean_distance(u, v):
    """
    计算两个向量的欧氏距离
    """
    return np.linalg.norm(u - v)

def clustering_accuracy(gold_labels: ArrayLike,gold_centers, clustered_labels: ArrayLike,cluster_centers):
    '''
    clusterd label -> cluster center -> nearest center == gold label?
    :param gold_labels:
    :param gold_centers:
    :param clustered_labels:
    :param cluster_centers:
    :return:
    '''
    if len(gold_labels)!= len(clustered_labels):
        raise ValueError('The length of gold labels must be equal to the length of cluster labels.')

    accuracy = 0
    for i in range(len(clustered_labels)):
        label = clustered_labels[i]
        center = cluster_centers[label]
        nearest = np.argmin([euclidean_distance(center,v) for v in gold_centers])
        gold_label = gold_labels[i]

        if nearest == gold_label:
            accuracy += 1
    return accuracy/len(clustered_labels)
