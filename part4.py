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

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(data, linkage_type, n_clusters):
    X, _ = data
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Perform hierarchical clustering
    clusterer = AgglomerativeClustering(linkage = linkage_type, n_clusters = n_clusters)
    clusterer.fit(X_scaled)
    # Return the labels
    return clusterer.labels_


def fit_modified(data, linkage_type):
    Z = linkage(data[0], method = linkage_type)
    distances = Z[:, 2]
    max_rate_of_change = np.max(np.diff(distances))
    cutoff_distance = distances[np.argmax(np.diff(distances))] + max_rate_of_change/2
    n_clusters = np.sum(distances >= cutoff_distance) + 1  # Number of clusters is one more than number of merges above cutoff
    labels = fit_hierarchical_cluster(data, linkage_type, n_clusters)
    return labels, cutoff_distance


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """
    n_samples = 100
    seed = 42
    # noisy-circles
    noisy_circles = datasets.make_circles(n_samples = n_samples, factor = 0.5, noise = 0.05, random_state = seed)

    # noisy-moons
    noisy_moons = datasets.make_moons(n_samples = n_samples, noise = 0.05, random_state = seed)

    # blobs with varied variances
    blobs_varied = datasets.make_blobs(n_samples = n_samples, cluster_std = [1.0, 2.5, 0.5], random_state = seed)

    # Anisotropicly distributed data
    X, y = datasets.make_blobs(n_samples = n_samples, random_state = seed)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_anisotrop = np.dot(X, transformation)
    anisotrop = (X_anisotrop, y)

    # blobs 
    blobs_normal = datasets.make_blobs(n_samples = n_samples, random_state = seed)

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {}
    dct["nc"] = noisy_circles
    dct["nm"] = noisy_moons
    dct["bvv"] = blobs_varied
    dct["add"] = anisotrop
    dct["b"] = blobs_normal
    

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """

    linkage_types = ['single', 'complete', 'ward', 'average']
    datasets_list = [noisy_circles, noisy_moons, blobs_varied, anisotrop, blobs_normal]
    dataset_names = ['Noisy Circles', 'Noisy Moons', 'Blobs with Varied Variances', 'Anisotropic', 'Blobs Normal']

    fig, axes = plt.subplots(nrows = len(linkage_types), ncols = len(datasets_list), figsize = (22, 16))

    for i, linkage_type in enumerate(linkage_types):
        for j, dataset in enumerate(datasets_list):
            labels = fit_hierarchical_cluster(dataset, linkage_type, 2)
            axes[i, j].scatter(dataset[0][:, 0], dataset[0][:, 1], c = labels, s = 8, cmap = 'viridis')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(dataset_names[j])
            if j == 0:
                axes[i, j].set_ylabel(linkage_type)

    fig.suptitle('Part 4B - Hierarchical Clustering with Different Linkage Types', fontsize = 15)
    plt.show()


    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["add","b"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """
    linkage_types = ['single', 'complete', 'ward', 'average']
    datasets_list = [noisy_circles, noisy_moons, blobs_varied, anisotrop, blobs_normal]
    dataset_names = ['Noisy Circles', 'Noisy Moons', 'Blobs with Varied Variances', 'Anisotropic', 'Blobs Normal']

    fig, axes = plt.subplots(nrows = len(linkage_types), ncols = len(datasets_list), figsize = (22, 16))
    
    for i, linkage_type in enumerate(linkage_types):
        for j, dataset in enumerate(datasets_list):
            labels, cutoff = fit_modified(dataset, linkage_type)
            axes[i, j].scatter(dataset[0][:, 0], dataset[0][:, 1], c = labels, s = 8, cmap = 'viridis')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(dataset_names[j])
            if j == 0:
                axes[i, j].set_ylabel(linkage_type)

    fig.suptitle('Part 4C - Hierarchical Clustering with Cut-off Distance', fontsize=15)
    plt.show()
    # dct is the function described above in 4.C
    dct = answers["4C: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
