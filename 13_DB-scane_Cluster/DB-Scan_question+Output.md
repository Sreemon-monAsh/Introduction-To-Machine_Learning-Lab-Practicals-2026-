# Program 13: DBSCAN Clustering

## Question 13
Implement DBSCAN (Density-Based Spatial Clustering of Applications with Noise) and evaluate clustering performance.

---

## 📂 Dataset

The dataset contains 2D points with:

- Two dense clusters
- One isolated noise point

Example structure:

X = [
    [1, 2], [2, 2], [2, 3], [1.5, 2.5], [1.8, 2.8], [1.2, 2.1],
    [8, 7], [8, 8], [8.5, 7.5], [7.8, 8.2], [8.2, 8.1], [8.1, 7.9],
    [50, 50]
]

## Output
    Cluster labels: [ 0  0  0  0  0  0  1  1  1  1  1  1 -1]
    Silhouette Score: 0.8446604515048153
    
![](db_scan_output.png)
