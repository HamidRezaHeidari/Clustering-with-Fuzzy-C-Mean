# import necessary libraries
from sklearn.datasets import load_iris
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score

# load dataset
iris = load_iris()
X = iris.data
y = iris.target


# define distance function
def distance(point1, point2):
    dis = ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2 + (
                point1[3] - point2[3]) ** 2) ** 0.5

    return dis


# define initial Membership Matrix (MvM) by normalized random values
def initial_membership_values(n_cluster):

    memeber_value_matrix=list()

    for i in range(150):
        random_list=[random.random() for x in range(n_cluster)]
        summation=sum(random_list)
        for i in range(len(random_list)):
            random_list[i]=random_list[i]/summation
        memeber_value_matrix.append(random_list)
    return memeber_value_matrix



# define center of clusters (CC) & membership matrix update function
def fuzzy_c_mean(memeber_value_matrix, n_cluster, m):

# Find Center of Clusters (CC)
    center_matrix = {}

    for j in range(n_cluster):
        center = []
        for k in range(4):

            b = 0
            for i in range(150):
                b += (memeber_value_matrix[i][j] ** m) * (X[i][k])
            a = 0
            for i in range(150):
                a += memeber_value_matrix[i][j] ** m
            center.append(b / a)
        center_matrix[j] = center

# Update the Membership values Matrix (MvM)
    for i in range(150):
        distances = list()
        for j in range(n_cluster):
            distances.append(distance(center_matrix[j], X[i]))
        for j in range(n_cluster):
            sigma = 0
            for k in range(n_cluster):
                sigma = sigma + (math.pow(distances[j] / distances[k], 2 / (m - 1)))
            memeber_value_matrix[i][j] = (1 / sigma)

    return memeber_value_matrix, center_matrix


# Start Clustering
n_cluster = 3 # number of Clusters
m = 2 # Constant Coefficient in Fuzzy_C_Mean
iteration = 100   # max iteration

#initial MvM
matrix = initial_membership_values(n_cluster) #initial MvM

for o in range(iteration):
    matrix, center = fuzzy_c_mean(matrix, n_cluster, m)
print(" final clusters center  : ", center)

# calculate summation of maximum distance of points from CC
dis_sum = 0
dis_from_centers = list()
for i in range(150):
    dis = list()
    for j in range(n_cluster):
        dis.append(distance(X[i], center[j]))
    dis_from_centers.append(dis)
    dis_sum += sum(dis)

# creat label of minimum distance of points from CC
labels = list()
for i in range(150):
    labels.append(np.argmin(dis_from_centers[i]))

# evaluate model
chs = calinski_harabasz_score(X, labels)

# Print the results
print("Score is : " ,chs)
print("distance summation is : " ,dis_sum)


# Plot the results
fig1 = plt.figure()
plt.scatter (X[:,0],X[:,1], c=labels)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
for i in range(n_cluster):
    plt.scatter(center[i][0], center[i][1], s=80, color="r", marker="D")

fig2 = plt.figure()
plt.scatter (X[:,2],X[:,3], c=labels)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
for i in range(n_cluster):
    plt.scatter(center[i][2], center[i][3], s=80, color="r", marker="D")

plt.show()