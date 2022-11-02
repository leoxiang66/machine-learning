from sklearn.metrics import silhouette_score
from ._clustering import BaseClustering
from typing import Type
from numpy import argmax

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



