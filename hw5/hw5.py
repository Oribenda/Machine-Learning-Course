import numpy as np

def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []

    centroids = X[np.random.choice(X.shape[0], k, replace=False), :]
    
    while np.unique(centroids).shape[0] < k:
        centroids = X[np.random.choice(X.shape[0], k, replace=False), :]
    
    return np.asarray(centroids).astype(np.float)

def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)

    X_as_row = X.reshape((1, X.shape[0], 3))
    centroids_as_col = centroids.reshape((centroids.shape[0], 1, 3))

    distances = np.sum(np.abs(X_as_row - centroids_as_col) ** p, axis=2)
    distances = distances ** (1 / p)

    return distances

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)

    for i in range(max_iter):
        distances = lp_distance(X, centroids, p)
        min_centroid = np.argmin(distances, axis=0)

        new_centroids = np.empty((k, 3))
        
        for j in range(k):
            new_centroids[j] = np.mean(X[min_centroid == j], axis=0)

        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids
        
    classes = min_centroid

    return centroids, classes

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = None

    centroids = get_random_centroids(X, 1)
    X_copy = np.copy(X)
    rows_to_delete = np.any(X_copy == np.array(centroids[0]).reshape(1, -1), axis=1)
    X_copy = X_copy[~rows_to_delete]
    
    while(centroids.shape[0] < k):
        distance = lp_distance(X_copy, centroids, p)
        distance = np.min(distance, axis=0)
        distance = np.square(distance)
        distance = distance / np.sum(distance)
        index = np.random.choice(X_copy.shape[0], 1, p=distance)
        new_centroid = X_copy[index, :]
        rows_to_delete = np.any(X_copy == np.array(new_centroid).reshape(1, -1), axis=1)
        X_copy = X_copy[~rows_to_delete]
        centroids = np.vstack((centroids, new_centroid))
        
    for i in range(max_iter):
        distances = lp_distance(X, centroids, p)
        min_centroid = np.argmin(distances, axis=0)

        new_centroids = np.empty((k, 3))
        
        for j in range(k):
            new_centroids[j] = np.mean(X[min_centroid == j], axis=0)

        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids
        
    classes = min_centroid

    return centroids, classes
