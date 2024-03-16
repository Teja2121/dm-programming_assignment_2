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
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data, n_clusters):
    X, _ = data

    scalar = StandardScaler()
    X_scalar = scalar.fit_transform(X)
    #k-means
    kmeans = cluster.KMeans(n_clusters = n_clusters, init = 'random', random_state = 80, n_init = 10)
    kmeans.fit(X_scalar)
    # label prediction
    labels = kmeans.predict(X_scalar)

    return labels


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """
    n_samples = 100
    seed = 42
    # noisy-circles
    noisy_circles = datasets.make_circles(n_samples = n_samples, factor = 0.5, noise = 0.05, random_state = seed)

    # noisy-moons
    noisy_moons = datasets.make_moons(n_samples = n_samples, noise = 0.05, random_state = seed)

    # blobs with varied variances
    blobs_varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state = seed)

    # Anisotropicly distributed data
    X, y = datasets.make_blobs(n_samples = n_samples, random_state = seed)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_anisotrop = np.dot(X, transformation)
    anisotrop = (X_anisotrop, y)

    # blobs 
    blobs_normal = datasets.make_blobs(n_samples = n_samples, random_state = seed)

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["1A: datasets"] = {}
    dct["nc"] = [noisy_circles, 'Noisy circles']
    dct["nm"] = [noisy_moons, 'Noisy moons']
    dct["bvv"] = [blobs_varied, 'Blobs with varied variance']
    dct["add"] = [anisotrop, 'Anisotropicly distributed data ']
    dct["b"] = [blobs_normal, 'Blobs']
    

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """

    datasets_list = [noisy_circles, noisy_moons, blobs_varied, anisotrop, blobs_normal]
    dataset_names = ['Noisy Circles', 'Noisy Moons', 'Blobs with Varied Variances', 'Anisotropic', 'Blobs Normal']

    # Defining the total number of clusters with the n_clusters_list 
    n_clusters_list = [2, 3, 5, 10]

    # Create a figure with subplots in a 4x5 grid
    fig, axis = plt.subplots(nrows = 4, ncols = 5, figsize = (22, 16))


    for i, n_clusters in enumerate(n_clusters_list):
        for j, dataset in enumerate(datasets_list):
            labels = fit_kmeans(dataset, n_clusters)
            axis[i, j].scatter(dataset[0][:, 0], dataset[0][:, 1], c = labels, s = 8, cmap = 'viridis')
            axis[i, j].set_xticks([])
            axis[i, j].set_yticks([])
            if i == 0:
                axis[i, j].set_title(dataset_names[j])
            if j == 0:
                axis[i, j].set_ylabel(f"k={n_clusters}")

    fig.suptitle('Part 1 - Plots', fontsize = 15) 
    plt.show()

    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    dct = answers["1C: cluster successes"] = {"nm": [10], "bvv": [2,10], "add" : [2, 10], "b" : [2,3,5,10]} 

    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)
    dct = answers["1C: cluster failures"] = ["nc"]

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.
    dct = answers["1D: datasets sensitive to initialization"] = ["nm"]

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
