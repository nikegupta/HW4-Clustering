from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters)
import numpy as np

# Write your k-means unit tests here
def test_kmeans(): 
    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    assert len(pred) == len(labels)

def test_num_points_error():
    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=1000)

    fail_flag = False
    try:
        km.fit(clusters)
    except ValueError:
        fail_flag = True
    else:
        pass

    assert fail_flag == True

def test_num_labels_error():
    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=4)
    km.fit(clusters)

    test_data = np.full((200,2), 0.)

    fail_flag = False
    try:
        km.predict(test_data)
    except ValueError:
        fail_flag = True
    else:
        pass

    assert fail_flag == True