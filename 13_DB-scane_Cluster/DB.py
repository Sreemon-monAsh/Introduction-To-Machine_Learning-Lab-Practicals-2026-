import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

X = np.array([
    [1, 2], [2, 2], [2, 3], [1.5, 2.5], [1.8, 2.8], [1.2, 2.1],
    [8, 7], [8, 8], [8.5, 7.5], [7.8, 8.2], [8.2, 8.1], [8.1, 7.9],
    [50, 50]   
])

dbscan = DBSCAN(eps=3, min_samples=2)

dbscan.fit(X)

labels = dbscan.labels_
print("Cluster labels:", labels)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', s=100)
plt.title("DBSCAN Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

score = silhouette_score(X, labels)
print("Silhouette Score:", score)
