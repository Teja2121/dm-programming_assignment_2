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
Part 3.	
Hierarchical Clustering: 
Recall from lecture that agglomerative hierarchical clustering is a greedy iterative scheme that creates clusters, i.e., distinct sets of indices of points, by gradually merging the sets based on some cluster dissimilarity (distance) measure. Since each iteration merges a set of indices there are at most n-1 mergers until the all the data points are merged into a single cluster (assuming n is the total points). This merging process of the sets of indices can be illustrated by a tree diagram called a dendrogram. Hence, agglomerative hierarchal clustering can be simply defined as a function that takes in a set of points and outputs the dendrogram.
"""

# Fill this function with code at this location. Do NOT move it.
# Change the arguments and return according to
# the question asked.


def data_index_function(data, I, J):
    # min_dist is being initialized
        min_dist = float('inf')
    
    # Computing the pariwise distances from the sets provided in I and J
        for i in I:
            for j in J:
                # Eucledian distance is calculated
                dist = np.sqrt(np.sum((data[i] - data[j]) ** 2))
                # updating the min_dist if it is smaller
                if dist < min_dist:
                    min_dist = dist

        return min_dist



def compute():
    answers = {}

    """
    A.	Load the provided dataset “hierachal_toy_data.mat” using the scipy.io.loadmat function.
    """
    toy_mat = io.loadmat('D:\\UNI\\Masters\\FSU\\Semester 2\\Data Mining\\Assignments\\Coding Assignment\\2\\hierarchical_toy_data.mat') 
    print(toy_mat.keys())
    data = toy_mat['X']

    print(data)

    # return value of scipy.io.loadmat()
    answers["3A: toy data"] = data

    """
    B.	Create a linkage matrix Z, and plot a dendrogram using the scipy.hierarchy.linkage and scipy.hierachy.dendrogram functions, with “single” linkage.
    """
    Z = linkage(data, method='single')

    print(f"The linkage matrix is: {Z}")

    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram (Single linkage)')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dend = dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
    plt.show()
    # Answer: NDArray
    answers["3B: linkage"] = Z

    # Answer: the return value of the dendogram function, dicitonary
    answers["3B: dendogram"] = dend

    """
    C.	Consider the merger of the cluster corresponding to points with index sets {I={8,2,13}} J={1,9}}. At what iteration (starting from 0) were these clusters merged? That is, what row does the merger of A correspond to in the linkage matrix Z? The rows count from 0. 
    """
    # Given a linkage matrix Z
    # Find the iteration where clusters I and J were merged
    I = {8, 2, 13}
    J = {1, 9}

    # Convert the sets to lists and sort them because the linkage matrix uses sorted indices
    I = sorted(list(I))
    J = sorted(list(J))

    # Initialize variables to store the merge iteration for each set
    merge_iteration_I = None
    merge_iteration_J = None

    # Iterate over linkage matrix to find the merge iteration for each set
    for index, row in enumerate(Z):
        cluster_1 = int(row[0])
        cluster_2 = int(row[1])

    # Since the linkage matrix contains n original points followed by merges,
    # we subtract n to find the merge iteration for clusters formed after the first n merges
        n = len(data)
        if cluster_1 >= n:
            cluster_1 -= n
        if cluster_2 >= n:
            cluster_2 -= n
    
        # Check if this row represents the merger of cluster I
        if cluster_1 in I or cluster_2 in I:
            merge_iteration_I = index
        # Check if this row represents the merger of cluster J
        if cluster_1 in J or cluster_2 in J:
            merge_iteration_J = index

    # If both mergers have been found, break out of the loop
        if merge_iteration_I is not None and merge_iteration_J is not None:
            break

    # Find out when the two clusters were merged together
    final_merge_iteration = max(merge_iteration_I, merge_iteration_J)

    # Output the result
    print("Clusters I and J were merged at iteration:", final_merge_iteration)

    # Answer type: integer
    answers["3C: iteration"] = final_merge_iteration 

    """
    D.	Write a function that takes the data and the two index sets {I,J} above, and returns the dissimilarity given by single link clustering using the Euclidian distance metric. The function should output the same value as the 3rd column of the row found in problem 2.C.
    """
    I = [8, 2, 13]
    J = [1, 9]
    dissimilarity_index = data_index_function(data, I, J)
    print(f"The dissimilarity is: {dissimilarity_index}")
    # Answer type: a function defined above
    answers["3D: function"] = data_index_function

    """
    E.	In the actual algorithm, deciding which clusters to merge should consider all of the available clusters at each iteration. List all the clusters as index sets, using a list of lists, 
    e.g., [{0,1,2},{3,4},{5},{6},…],  that were available when the two clusters in part 2.D were merged.
    """

    # List the clusters. the [{0,1,2}, {3,4}, {5}, {6}, ...] represents a list of lists.
    answers["3E: clusters"] = [[0, 0], [0, 0]]

    """
    F.	Single linked clustering is often criticized as producing clusters where “the rich get richer”, that is, where one cluster is continuously merging with all available points. Does your dendrogram illustrate this phenomenon?
    """

    # Answer type: string. Insert your explanation as a string.
    answers["3F: rich get richer"] = "True, this does happen in the dendogram where the clusters are merging."

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part3.pkl", "wb") as f:
        pickle.dump(answers, f)
