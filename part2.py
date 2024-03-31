from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data, n_clusters):
    X, _ = data
    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)
    kmeans = cluster.KMeans(n_clusters=n_clusters, init='random', n_init = 10)
    kmeans.fit(X_scaled)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    sse = 0
    for i in range(n_clusters):
        cluster_points = X_scaled[labels == i]
        centroid = centroids[i]
        sse += np.sum((cluster_points - centroid) ** 2)
    return sse



def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """
    blob_data = make_blobs(center_box=(-20, 20), n_samples = 20, centers = 5, random_state = 12)
    X, y = blob_data
    centers = np.unique(y)
    list_1A = [X, y, centers]

    print("Part 2A : \n")
    for value2 in list_1A:
        print(value2)

    print()


    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    dct = answers["2A: blob"] = [X, y, centers]


    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """
    

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """

    sse_values = []
    for k in range(1, 9):
        sse_1 = fit_kmeans(blob_data, k)
        sse_values.append((sse_1))

    print("Part 2C : \n")

    value_list_1 = [[i+1, value_list] for i, value_list in enumerate(sse_values)]
    print(value_list_1)

    print()

    plt.figure(figsize = (10, 6))
    plt.plot(range(1, 9), sse_values, 'o-', color = 'orange')
    plt.xlabel('Number of clusters k')
    plt.ylabel('SSE')
    plt.title('Part 2C - Elbow Method')
    plt.show()    

    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    dct = answers["2C: SSE plot"] = value_list_1

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """
    scalar_std = StandardScaler()
    X_scaled_1 = scalar_std.fit_transform(X)

    wcss_values = []
    for k in range(1, 9):
        sse_1 = cluster.KMeans(n_clusters = k, n_init = 10)
        sse_1.fit(X_scaled_1)
        wcss_values.append((sse_1.inertia_))

    print("Part 2D : \n")

    value_list_2 = [[j+1, value_list] for j, value_list in enumerate(wcss_values)]
    print(value_list_2)

    print()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 9), wcss_values, 'o-')
    plt.xlabel('Number of clusters k')
    plt.ylabel('WCSS (Within cluster sum of squares )')
    plt.title('Part 2D - Inertia')
    plt.show() 

    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = value_list_2

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
