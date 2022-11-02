if __name__ == '__main__':
    from tqdm import tqdm
    from sklearn.datasets import make_blobs
    from unsupervised_learning.clustering import KMeans,GaussianMixture, Silhouette
    import matplotlib.pyplot as plt


    # Create dataset with 5 random cluster centers and 1000 datapoints
    # x, y = make_blobs(n_samples=1000, centers=5, n_features=2, shuffle=True, random_state=31)
    #
    # print(Silhouette(GaussianMixture,2,10,0).get_best_k(x))
    # range_ = range(1,769,100)
    # ns = list(range_)
    # ks = []
    # metric = Silhouette(GaussianMixture,2,10,0)
    # for n_feature in tqdm(range_):
    #     x, y = make_blobs(n_samples=1000, centers=5, n_features=n_feature, shuffle=True, random_state=31)
    #     ks.append(metric.get_best_k(x))

    # plt.plot(ns,ks)
    # plt.show()
    
    '''
    features从1->768, best k 都等于5, 说明features数量对 $k$ 影响不大
    '''
    range_ = range(100,2000,100)
    ns = list(range_)
    ks = []
    metric = Silhouette(GaussianMixture,2,10,0)
    for n_samples in tqdm(range_):
        x, y = make_blobs(n_samples=n_samples, centers=5, n_features=500, shuffle=True, random_state=31)
        ks.append(metric.get_best_k(x))

    plt.plot(ns,ks)
    plt.show()
    


