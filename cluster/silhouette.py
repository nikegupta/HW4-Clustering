import numpy as np

class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

            k: int
                number of clusters

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        num_points = X.shape[0]

        #silhouette score array
        silhouette_scores = np.full(num_points, np.nan)

        #calculate number of points in each cluster
        num_per_cluster = np.full(k, 0.)
        for i in range(num_points):
            num_per_cluster[y[i]] += 1

        for i in range(num_points):

            #initialize lists distance scores to other clusters
            distance_scores_sum = np.full(k, 0.)
            distance_scores_average = np.full(k,0.)

            #remove your point from cluster totals
            num_per_cluster_i = num_per_cluster
            num_per_cluster_i[y[i]] -= 1


            for j in range(num_points):

                #exclude your point from tabulation
                if j != i:

                    #add distance score to point j
                    distance_scores_sum[y[j]] += np.linalg.norm(X[i] - X[j])

            #average distance scores
            for j in range(k):
                distance_scores_average[j] = ( distance_scores_sum[j] / num_per_cluster_i[j] )

            #get parts for silhouette score calculation
            a = distance_scores_average[y[i]]
            other_cluster_scores = []
            for j in range(k):
                if j != y[i]:
                    other_cluster_scores.append(distance_scores_average[j])
            b = min(other_cluster_scores)

            #calculate silhouette score
            silhouette_scores[i] = ( b - a ) / max([a,b])

            #add point back
            num_per_cluster_i[y[i]] += 1

        return silhouette_scores



            

