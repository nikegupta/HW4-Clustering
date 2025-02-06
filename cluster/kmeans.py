import numpy as np
import random


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        #initialize random centers
        num_points = mat.shape[0]
        #check k > num_points
        if num_points < self.k:
            raise ValueError('Number of points smaller than cluster')

        self.fit_data = mat
        self.centroids = [mat[i, :] for i in random.sample(range(1,num_points), self.k)]

        #main loop
        flag = True
        while flag:

            #make empty label array
            labels = np.full(num_points, 0, dtype=int)

            #for each point find closest center
            for n_i in range(num_points):
                best_distance = float('inf')
                for c_i in range(len(self.centroids)):
                    distance = np.linalg.norm(mat[n_i] - self.centroids[c_i])

                    #assign that point to that cluster
                    if distance < best_distance:
                        best_distance = distance
                        labels[n_i] = c_i

            #store label information
            self.labels = labels

            #recalculate centroids
            self.centroids = self._get_centroids(mat)

            #calculate change in centroids and break loop if smaller than tolerance
            if self.change < self.tol:
                flag = False
                
        

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        num_points = mat.shape[0]
        labels = np.full(num_points, 0, dtype=int)

        if mat.shape != self.fit_data.shape:
            raise ValueError('Input Matrix Difference Size that Fit Data')
        
        for n_i in range(num_points):
            best_distance = float('inf')
            for c_i in range(len(self.centroids)):
                distance = np.linalg.norm(mat[n_i] - self.centroids[c_i])

                if distance < best_distance:
                        best_distance = distance
                        labels[n_i] = c_i

        return labels

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """

        error = 0
        num_points = self.fit_data.shape[0]
        for i in range(num_points):
            error += np.linalg.norm(self.fit_data[i] - self.centroids[self.labels[i]])

        return error

    def _get_centroids(self, mat) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            new_centroids: np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        #variable for biggest change in centroid
        self.change = 0

        #iterate through centroids
        new_centroids = []
        for c_i in range(len(self.centroids)):

            #get all points in that cluster
            cluster_points = []
            for i in range(len(self.labels)):
                if self.labels[i] == c_i:
                    cluster_points.append(mat[i, :])
            array = np.array(cluster_points, dtype=float)

            #take average of points
            new_centroids.append(np.average(array, axis=0))

            difference = np.linalg.norm(new_centroids[c_i] - self.centroids[c_i])
            if difference > self.change:
                self.change = difference

        return new_centroids





