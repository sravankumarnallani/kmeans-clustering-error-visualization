import math
import random
import matplotlib.pyplot as plt
import sys

def cdy_load_data(cdy_file_path):
    """ 
    Load data from a file, ignoring the last column (class labels).
    
    This function reads a whitespace-delimited text file where each line is a data record.
    The last value in each line, presumed to be a class label, is ignored, making this function
    versatile for datasets where the class label is appended at the end.
    
    Parameters:
        cdy_file_path (str): The path to the dataset file.

    Returns:
        list: A list of lists, with each sublist containing floating point numbers representing the features of one data point.
    """
    with open(cdy_file_path, 'r') as f:
        cdy_data = [list(map(float, line.split()[:-1])) for line in f]
    return cdy_data

def cdy_initialize_centroids(cdy_data, cdy_k):
    """
    Initialize centroids by randomly selecting 'k' points from the dataset.
    
    This approach ensures a simple random selection of initial centroids, which is a common
    method in k-means clustering algorithms but can lead to suboptimal clustering results
    if unlucky picks are made. The seed for the random number generator is set to ensure
    reproducibility of results.

    Parameters:
        cdy_data (list of lists): The dataset from which to select centroids.
        cdy_k (int): The number of centroids to initialize.

    Returns:
        list: A list of 'k' lists, each being a centroid initialized to one of the points randomly picked from the data.
    """
    random.seed(0)
    cdy_centroids = random.sample(cdy_data, cdy_k)
    return cdy_centroids

def cdy_assign_clusters(cdy_data, cdy_centroids):
    """
    Assign each data point to the nearest centroid based on Euclidean distance.
    
    This function calculates the Euclidean distance from each data point to each centroid
    and assigns the point to the centroid with the minimum distance. This step is key in k-means
    for forming clusters based on nearest mean similarity.

    Parameters:
        cdy_data (list of lists): The dataset.
        cdy_centroids (list of lists): Current centroids.

    Returns:
        list: A list of indices where each index corresponds to the centroid assigned to each data point.
    """
    cdy_assignments = []
    for point in cdy_data:
        distances = [math.sqrt(sum((px - cx) ** 2 for px, cx in zip(point, centroid))) for centroid in cdy_centroids]
        closest_centroid_index = distances.index(min(distances))
        cdy_assignments.append(closest_centroid_index)
    return cdy_assignments

def cdy_update_centroids(cdy_data, cdy_assignments, cdy_k):
    """
    Update centroids by calculating the mean of assigned data points for each cluster.

    After all data points have been assigned to clusters, this function recalculates each centroid as
    the mean of all points assigned to it. This step is crucial for moving the centroids toward the
    'center' of their respective clusters.

    Parameters:
        cdy_data (list of lists): The dataset.
        cdy_assignments (list): A list of centroid indices assigned to each data point.
        cdy_k (int): The number of clusters.

    Returns:
        list: A list of updated centroids calculated as the mean of data points assigned to each cluster.
    """
    new_centroids = []
    for i in range(cdy_k):
        assigned_points = [p for p, a in zip(cdy_data, cdy_assignments) if a == i]
        if assigned_points:
            centroid = [sum(p[j] for p in assigned_points) / len(assigned_points) for j in range(len(cdy_data[0]))]
            new_centroids.append(centroid)
    return new_centroids

def cdy_calculate_error(cdy_data, cdy_centroids, cdy_assignments):
    """
    Calculate the clustering error as the sum of squared distances from points to their centroids.
    
    This function sums the squared Euclidean distances from each data point to the centroid
    it has been assigned to. This metric helps in evaluating the compactness of the clusters formed,
    which is a key measure of clustering quality.

    Parameters:
        cdy_data (list of lists): The dataset.
        cdy_assignments (list): The index of the centroid assigned to each data point.
        cdy_centroids (list of lists): The current centroids.

    Returns:
        float: The total sum of squared distances from each point to its assigned centroid.
    """
    total_distance = 0
    for point, assignment in zip(cdy_data, cdy_assignments):
        distance = math.sqrt(sum((px - cx) ** 2 for px, cx in zip(point, cdy_centroids[assignment])))
        total_distance += distance
    return total_distance

def cdy_k_means_clustering(cdy_data, cdy_k):
    """
    Implement the k-means clustering algorithm.
    
    This function orchestrates the entire k-means clustering process, which includes initializing centroids,
    assigning clusters, updating centroids, and calculating the clustering error after a fixed number of iterations.

    Parameters:
        cdy_data (list of lists): Data to cluster.
        cdy_k (int): Number of clusters.

    Returns:
        float: The clustering error after the final iteration.
    """
    cdy_centroids = cdy_initialize_centroids(cdy_data, cdy_k)
    for _ in range(20):  # Perform 20 iterations
        cdy_assignments = cdy_assign_clusters(cdy_data, cdy_centroids)
        new_centroids = cdy_update_centroids(cdy_data, cdy_assignments, cdy_k)
        if cdy_centroids == new_centroids:
            break
        cdy_centroids = new_centroids
    cdy_error = cdy_calculate_error(cdy_data, cdy_centroids, cdy_assignments)
    return cdy_error

def cdy_plot_errors(cdy_ks, cdy_errors):
    """
    Plot the error values against k values.
    
    This function generates a plot of clustering error versus the number of clusters,
    which is helpful for visualizing the performance of the k-means algorithm across different numbers of clusters.
    
    Parameters:
        cdy_ks (range): The range of k values tested.
        cdy_errors (list): The errors associated with each k value.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(cdy_ks, cdy_errors, marker='o')
    plt.title('Error vs. Number of Clusters')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Error')
    plt.xticks(cdy_ks)
    plt.grid(True)
    plt.show()

def cdy_main(cdy_data_file):
    """
    Load data, run k-means for k=2 to k-10, and plot errors.
    
    This is the main function that sets up the data, runs the k-means clustering algorithm for a range of k values,
    records the errors, and finally plots these errors to assess the performance of the clustering.

    Parameters:
        cdy_data_file (str): Path to the dataset file.
    """
    cdy_data = cdy_load_data(cdy_data_file)
    cdy_ks = range(2, 11)
    cdy_errors = []
    
    for k in cdy_ks:
        error = cdy_k_means_clustering(cdy_data, k)
        cdy_errors.append(error)
        print(f"For k = {k} After 20 iterations: Error = {error:.4f}")
    
    cdy_plot_errors(cdy_ks, cdy_errors)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python kmeans.py <data_file>")
    else:
        cdy_main(sys.argv[1])
