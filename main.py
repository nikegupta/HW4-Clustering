import numpy as np
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)
from sklearn.metrics import silhouette_score

def main():
    # create loose clusters
    clusters, labels = make_clusters(scale=2)
    plot_clusters(clusters, labels, filename="figures/loose_clusters.png")

    """
    uncomment this section once you are ready to visualize your kmeans + silhouette implementation
    """
    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred, 4)
    sk_scores = silhouette_score(clusters,pred)
    average_score = np.mean(scores)
    plot_multipanel(clusters, labels, pred, scores)
    error = km.get_error()
    print(error)
    

if __name__ == "__main__":
    main()
