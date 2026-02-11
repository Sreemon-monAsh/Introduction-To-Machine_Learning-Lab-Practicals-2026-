import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import seaborn as sn
import matplotlib.pyplot as plt

# columns = ['sepal-length', 'sepal-width',
#            'petal-length', 'petal-width', 'class']

df = pd.read_csv('iris.csv')

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=1
)

clf = SVC(kernel='linear')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Predicted Values:")
print(y_pred)

print("Accuracy is:", accuracy_score(y_test, y_pred) * 100)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

confusion_matrix = pd.crosstab(
    y_test,
    y_pred,
    rownames=['Actual'],
    colnames=['Predicted']
)

sn.heatmap(confusion_matrix, annot=True)
plt.show()
