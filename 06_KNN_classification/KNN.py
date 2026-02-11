import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("diabetes.csv")

X = df.iloc[:, 0:8].values
y = df.iloc[:, 8].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted values:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

error = []
max_k = len(X_train)

for k in range(1, max_k + 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred_k = knn.predict(X_test)
    error.append(np.mean(pred_k != y_test))

plt.figure(figsize=(10, 5))
plt.plot(range(1, max_k + 1), error, marker='o')
plt.title("Error Rate vs K Value")
plt.xlabel("K Value")
plt.ylabel("Mean Error")
plt.grid(True)
plt.show()
