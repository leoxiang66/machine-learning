import numpy as np
from sklearn.metrics import silhouette_score
from ._clustering import BaseClustering
from typing import Type
from numpy import argmax
from sklearn.mixture import GaussianMixture

class BaseBestK:
    def __init__(self, modeltype: Type[BaseClustering], k_min:int =2, k_max: int = 10, random_state = 0):
        if k_min >= k_max or k_min < 2:
            raise ValueError('Invalid values for k_min and k_max')
        self.__modeltype__ = modeltype
        self.__k_min__ = k_min
        self.__k_max__ = k_max
        self.__random_state__ = random_state

    def get_best_k(self,X,y=None,sample_weight = None):
        raise NotImplementedError()

class Silhouette(BaseBestK):
    def get_best_k(self, X, y=None, sample_weight = None):
        sil = []
        ks = range(self.__k_min__, self.__k_max__ + 1)
        for k in ks:
            model = self.__modeltype__(k,self.__random_state__)
            model.fit(X, y, sample_weight)
            labels = model.getLabels()
            sil.append(silhouette_score(X, labels, metric='euclidean'))

        return ks[argmax(sil)]


class BIC(BaseBestK):

    def get_best_k(self, X, y=None, sample_weight=None):
        # 初始化BIC和最佳聚类数量
        bic = []
        best_n_components = 0

        # 通过循环来计算不同聚类数量的BIC
        for n_components in range(1, 11):
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(X)
            bic.append(gmm.bic(X))

        # 找到最小BIC对应的聚类数量
        best_n_components = np.argmin(bic) + 1
        return best_n_components