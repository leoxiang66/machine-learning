from sklearn.cluster import KMeans as __KMeans_skl__
from sklearn.mixture import GaussianMixture as __GMM__

class BaseClustering:
    def __init__(self, k:int, random_state: int) -> None:
        super().__init__()
        self.BACKEND = None

    def fit(self, X, y = None, sample_weight = None):
        '''

        :param X: of shape N,D
        :return:  None
        '''
        raise NotImplementedError()

    def getLabels(self):
        '''

        :return: of shape N with entries in [0, 1, ..., k-1]
        '''
        raise NotImplementedError()

    def getClusterCenters(self):
        '''

        :return: of shape K,D
        '''
        raise NotImplementedError()


class KMeans(BaseClustering):
    def __init__(self, k: int, random_state: int) -> None:
        super().__init__(k, random_state)
        self.BACKEND = __KMeans_skl__(
            n_clusters=k,
            random_state = random_state
        )

    def fit(self, X, y = None, sample_weight = None):
        self.BACKEND.fit(X,y,sample_weight)

    def getLabels(self):
        return self.BACKEND.labels_

    def getClusterCenters(self):
        return self.BACKEND.cluster_centers_

class GaussianMixture(BaseClustering):
    def __init__(self, k: int, random_state: int) -> None:
        super().__init__(k, random_state)
        self.BACKEND = __GMM__(
            n_components= k,
            random_state= random_state
        )
        self.X = None

    def fit(self, X, y=None, sample_weight=None):
        self.BACKEND.fit(X,y)
        self.X = X

    def getLabels(self):
        return self.BACKEND.predict(self.X)

    def getClusterCenters(self):
        return self.BACKEND.means_

if __name__ == '__main__':
    from sklearn.datasets import make_blobs

    # Create dataset with 5 random cluster centers and 1000 datapoints
    x, y = make_blobs(n_samples=1000, centers=5, n_features=2, shuffle=True, random_state=31)


    kmeans = KMeans(3,0)
    kmeans.fit(x)
    print(len(kmeans.getLabels()))
    print(kmeans.getClusterCenters())


    gmm = GaussianMixture(3,0)
    gmm.fit(x)
    print(len(gmm.getLabels()))
    print(gmm.getLabels())
    print(gmm.getClusterCenters())