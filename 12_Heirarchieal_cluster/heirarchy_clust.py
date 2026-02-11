import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

X = np.array([
    [2, 8],
    [8, 15],
    [3, 6],
    [6, 9],
    [8, 7],
    [10, 10]
])

cluster = AgglomerativeClustering(
    n_clusters=3,
    metric='euclidean',
    linkage='ward'
)

labels = cluster.fit_predict(X)

print("Cluster Labels:")
print(labels)

plt.figure(figsize=(10, 7))
plt.title("Employee Skill Dendrograms")

dend = shc.dendrogram(
    shc.linkage(X, method='ward')
)

plt.show()
