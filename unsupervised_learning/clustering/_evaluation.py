from numpy.typing import ArrayLike
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score
import numpy as np
from prettytable import PrettyTable

def euclidean_distance(u, v):
    """
    计算两个向量的欧氏距离
    """
    return np.linalg.norm(u - v)

def evaluate_clustering(data: np.ndarray ,gold_labels: ArrayLike, pred_labels: ArrayLike, print_results:bool = True):
    # 计算评估指标
    ari = adjusted_rand_score(gold_labels, pred_labels)
    silhouette = silhouette_score(data, pred_labels)
    chi = calinski_harabasz_score(data, pred_labels)

    if print_results:
        table = PrettyTable()
        # Add columns to the table
        table.field_names = ["Adjusted Rand Index", "Silhouette Score", "Calinski-Harabasz Index"]
        table.add_row([ari,silhouette,chi])
        print(table)
        print()
    
    return dict(
        ARI = ari,
        SS = silhouette,
        CHI = chi
    )
    
    
# if __name__ == '__main__':
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from sklearn import datasets
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
#     from sklearn.mixture import GaussianMixture
#     from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score
#     data, labels_true = datasets.make_blobs(n_samples=300, centers=4, random_state=42)
#
#     # 数据预处理
#     scaler = StandardScaler()
#     data = scaler.fit_transform(data)
#
#     # 定义聚类算法
#     clustering_algorithms = [
#         KMeans(n_clusters=4),
#         DBSCAN(eps=0.3, min_samples=10),
#         AgglomerativeClustering(n_clusters=4),
#         SpectralClustering(n_clusters=4, assign_labels='discretize'),
#         GaussianMixture(n_components=4)
#     ]
#
#     # 评估聚类算法
#     for algorithm in clustering_algorithms:
#         # 对于GMM，使用predict方法，其他算法使用fit_predict
#         if isinstance(algorithm, GaussianMixture):
#             algorithm.fit(data)
#             labels_pred = algorithm.predict(data)
#         else:
#             labels_pred = algorithm.fit_predict(data)
#
#         evaluate_clustering(data,labels_true,labels_pred)