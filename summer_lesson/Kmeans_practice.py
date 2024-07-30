import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance
import time

## Generate the data
iris = datasets.load_iris()
iris_data = iris.data[:, :2]  # Only take the first two features for simplicity

## Randomly generate initial center points
K = 3
np.random.seed()
initial_centers = iris_data[np.random.choice(iris_data.shape[0], K, replace=False)]

centers = initial_centers
max_iters = 100


plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))

for iteration in range(max_iters):
    
    distances = distance.cdist(iris_data, centers, 'euclidean')
    labels = np.argmin(distances, axis=1)
    new_centers = np.array([iris_data[labels == i].mean(axis=0) for i in range(K)])
    
    ## Plot the clusters and centers
    ax.clear()
    colors = ['r', 'g', 'b']
    for i in range(K):
        points = iris_data[labels == i]
        ax.scatter(points[:, 0], points[:, 1], s=30, c=colors[i], label=f'Cluster {i}')
    ax.scatter(new_centers[:, 0], new_centers[:, 1], s=200, c='y', marker='X', edgecolor='k', label='Centers')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'Iteration {iteration + 1}')
    ax.legend()
    plt.draw()
    plt.pause(0.5)
    
    ## Check for convergence
    if np.all(centers == new_centers):
        print("Final centers:\n", centers)
        break
    centers = new_centers

# Turn off interactive mode
plt.ioff()

## Plot the final clusters
plt.figure(figsize=(8, 6))
for i in range(K):
    points = iris_data[labels == i]
    plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[i], label=f'Cluster {i}')
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='y', marker='X', edgecolor='k', label='Centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Final Clusters with K-Means')
plt.legend()
plt.show()
