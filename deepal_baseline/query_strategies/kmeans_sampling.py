import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans

class KMeansSampling(Strategy):
    def __init__(self, dataset, net, n_init=10):
        super(KMeansSampling, self).__init__(dataset, net)
        self.n_init = n_init

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data() # y is not masked like in scikit
        embeddings = self.get_embeddings(unlabeled_data)
        embeddings = embeddings.numpy()
        
        # Ensure the number of clusters doesn't exceed the number of available points
        n_clusters = min(n, embeddings.shape[0])
        cluster_learner = KMeans(n_clusters=n_clusters, n_init=self.n_init)
        
        cluster_learner.fit(embeddings)
        
        cluster_idxs = cluster_learner.predict(embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embeddings - centers)**2
        dis = dis.sum(axis=1)
        q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n_clusters)])

        return unlabeled_idxs[q_idxs]



