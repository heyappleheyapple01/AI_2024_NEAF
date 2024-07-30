import numpy as np
np.set_printoptions(threshold=np.inf)  ## print all values of matrix without reduction
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance     ## calculate the distance between two points
from sklearn.decomposition import PCA

## First part: load dataset (very famous called iris)
iris = datasets.load_iris()
iris_data = iris.data
iris_label = iris.target

## Second part: perform PCA to reduce dimensions to 2
pca = PCA(n_components=2)
iris_reduced = pca.fit_transform(iris_data)

## Third part: plot the distribution of data
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
color_map = [colors[label] for label in iris_label]
scatter = plt.scatter(iris_reduced[:, 0], iris_reduced[:, 1], c=color_map)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')

## Fourth part: the principle of KNN
test_point = np.array([[5.0, 3.6, 1.4, 0.2]])
test_reduced = pca.transform(test_point)

K = 5
class_num = 3
class_count = [0, 0, 0]
dst_array = []

for i in range(iris_label.shape[0]):
    dst = distance.euclidean(test_reduced[0], iris_reduced[i])
    dst_array.append(dst)

idx_sort = np.argsort(dst_array)[0:K]

for i in range(K):
    label = iris_label[idx_sort[i]]
    class_count[label] += 1

result = np.argsort(class_count)[-1]
print(f"Predicted class for the test point: {result}")

light_colors = ['lightcoral', 'lightgreen', 'lightblue']
test_color = light_colors[result]

plt.scatter(test_reduced[:, 0], test_reduced[:, 1], c=test_color, edgecolor='k', marker='X', s=100, label='Test Point')
plt.legend()
plt.show()
