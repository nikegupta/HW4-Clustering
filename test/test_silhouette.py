from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters)
from sklearn.metrics import silhouette_score
import numpy as np

# write your silhouette score unit tests here
def test_silhouette():
    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred, 4)
    sk_score = silhouette_score(clusters,pred)
    average_score = np.mean(scores)

    assert sk_score - average_score < 0.01