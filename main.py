"""
Fuzzy C-Means Clustering on Iris Dataset
This script implements a Fuzzy C-Means (FCM) clustering algorithm from scratch
and evaluates it using the Calinski-Harabasz score.
"""

# Import necessary libraries
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import calinski_harabasz_score

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # True labels (not used in clustering)

def euclidean_distance(point1, point2):
    """
    Compute the Euclidean distance between two points in 4D space.
    
    Args:
        point1 (list or np.array): Coordinates of the first point.
        point2 (list or np.array): Coordinates of the second point.
        
    Returns:
        float: Euclidean distance between the points.
    """
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def initialize_membership_matrix(n_samples, n_clusters):
    """
    Initialize membership matrix with normalized random values.
    
    Args:
        n_samples (int): Number of data points.
        n_clusters (int): Number of clusters.
        
    Returns:
        list: Membership value matrix (n_samples x n_clusters)
    """
    membership_matrix = []
    for _ in range(n_samples):
        random_values = [random.random() for _ in range(n_clusters)]
        total = sum(random_values)
        normalized_values = [val / total for val in random_values]
        membership_matrix.append(normalized_values)
    return membership_matrix

def fuzzy_c_means(membership_matrix, n_clusters, m):
    """
    Perform one iteration of Fuzzy C-Means:
    1. Calculate cluster centers
    2. Update membership matrix
    
    Args:
        membership_matrix (list): Current membership values.
        n_clusters (int): Number of clusters.
        m (float): Fuzziness coefficient (m > 1)
        
    Returns:
        tuple: Updated membership matrix and cluster centers
    """
    n_samples = len(membership_matrix)
    n_features = len(X[0])
    
    # Calculate cluster centers
    centers = {}
    for j in range(n_clusters):
        center = []
        for k in range(n_features):
            numerator = sum((membership_matrix[i][j] ** m) * X[i][k] for i in range(n_samples))
            denominator = sum(membership_matrix[i][j] ** m for i in range(n_samples))
            center.append(numerator / denominator)
        centers[j] = center

    # Update membership values
    for i in range(n_samples):
        distances = [euclidean_distance(centers[j], X[i]) for j in range(n_clusters)]
        for j in range(n_clusters):
            membership_matrix[i][j] = 1 / sum((distances[j] / distances[k]) ** (2 / (m - 1)) for k in range(n_clusters))
    
    return membership_matrix, centers

# -------------------
# Clustering Settings
# -------------------
n_clusters = 3
m = 2  # Fuzziness coefficient
max_iterations = 100
n_samples = X.shape[0]

# Initialize membership matrix
membership_matrix = initialize_membership_matrix(n_samples, n_clusters)

# Run Fuzzy C-Means iterations
for _ in range(max_iterations):
    membership_matrix, centers = fuzzy_c_means(membership_matrix, n_clusters, m)

print("Final cluster centers:", centers)

# Calculate distance sums for evaluation
distance_sums = []
for i in range(n_samples):
    distance_sums.append([euclidean_distance(X[i], centers[j]) for j in range(n_clusters)])
total_distance_sum = sum(sum(dist) for dist in distance_sums)

# Assign labels based on minimum distance to cluster centers
labels = [np.argmin(dist) for dist in distance_sums]

# Evaluate clustering performance
chs_score = calinski_harabasz_score(X, labels)
print("Calinski-Harabasz Score:", chs_score)
print("Total distance sum:", total_distance_sum)

# -------------------
# Plotting results
# -------------------
# Plot Sepal dimensions
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
for i in range(n_clusters):
    plt.scatter(centers[i][0], centers[i][1], s=80, color="red", marker="D")
plt.title("Fuzzy C-Means Clustering (Sepal)")

# Plot Petal dimensions
plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=labels, cmap='viridis')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
for i in range(n_clusters):
    plt.scatter(centers[i][2], centers[i][3], s=80, color="red", marker="D")
plt.title("Fuzzy C-Means Clustering (Petal)")

plt.show()
