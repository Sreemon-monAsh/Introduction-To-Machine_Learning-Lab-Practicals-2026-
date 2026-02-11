# Program 10: Support Vector Machine (SVM) – Linear Kernel

## Question 10
Implement Support Vector Machine (SVM) using a Linear Kernel to classify Iris flower species.

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
    Predicted Values:
    ['setosa' 'versicolor' 'versicolor' 'setosa' 'virginica' 'versicolor'
     'virginica' 'setosa' 'setosa' 'virginica' 'versicolor' 'setosa'
     'virginica' 'versicolor' 'versicolor' 'setosa' 'versicolor' 'versicolor'
     'setosa' 'setosa' 'versicolor' 'versicolor' 'versicolor' 'setosa'
     'virginica' 'versicolor' 'setosa' 'setosa' 'versicolor' 'virginica'
     'versicolor' 'virginica' 'versicolor' 'virginica' 'virginica' 'setosa'
     'versicolor' 'setosa' 'versicolor' 'virginica' 'virginica' 'setosa'
     'virginica' 'virginica' 'versicolor']
    Accuracy is: 100.0
    
    Classification Report:
                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        14
      versicolor       1.00      1.00      1.00        18
       virginica       1.00      1.00      1.00        13
    
        accuracy                           1.00        45
       macro avg       1.00      1.00      1.00        45
    weighted avg       1.00      1.00      1.00        45

![](svm_output.png)
