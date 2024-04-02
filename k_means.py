import numpy as np


def find_closest_centroids(X, centroids):
    """
    Finds the closest centroid for each example in X.
    Args:
        X         : array of data points, shape (m, n)
        centroids : array of centroids, shape (K, n)
    Returns:
        idx       : array of shape (m, ), where each entry is
                    in [0, K-1] corresponding to a centroid.
    """
    # Initialize values
    m = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros(m, dtype=int)

    for i in range(X.shape[0]):
        # calculate the distance
        dist = []
        for j in range(centroids.shape[0]):
            # calculate the distance between the data point and the centroid
            # the methods norm represents the Euclidean distance
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            dist.append(norm_ij)

        idx[i] = np.argmin(dist)  # what argmin does is to return the index of the minimum value in the array
    return idx


def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    centroids = np.zeros((K, n))

    ### START CODE HERE ###
    # sum the distance
    for i in m:
        # add X[i] to certain centroid
        for j in range(n):
            centroids[idx[i]][j] += X[i][j]
    # get average
    for i in range(K):
        for j in range(n):
            centroids[i][j] /= np.sum(idx == i)
    ### END CODE HERE ##

    return centroids
