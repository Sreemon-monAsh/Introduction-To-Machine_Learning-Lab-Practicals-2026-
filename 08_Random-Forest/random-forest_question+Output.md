# Program 08: Random Forest Classification (Iris Dataset)

## Question 8
Implement Random Forest Classification to predict the species of an Iris flower.

---

## 📂 Dataset

File: `iris.csv`

Columns:

1. sepal-length  
2. sepal-width  
3. petal-length  
4. petal-width  
5. class  

Target Classes:

- setosa  
- versicolor  
- virginica

## Output
    Accuracy: 97.77777777777777
    
    Classification Report:
                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        16
      versicolor       1.00      0.94      0.97        18
       virginica       0.92      1.00      0.96        11
    
        accuracy                           0.98        45
       macro avg       0.97      0.98      0.98        45
    weighted avg       0.98      0.98      0.98        45
    
    
    Confusion Matrix:
    [[16  0  0]
     [ 0 17  1]
 [ 0  0 11]]
