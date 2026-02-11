# Program 05: Naive Bayes Classification

## Question 5
Implement Naive Bayes Classification to predict whether a person will buy a computer based on:

- Age
- Income
- Student Status
- Credit Rating

---

## 📂 Dataset Description

This program uses manually defined categorical data.

Features:

- age → youth, middle-aged, senior
- income → high, medium, low
- student → Yes, No
- credit_rating → fair, excellent

Target:

- buys_computer → yes, no

## Output
    [2 2 0 1 1 1 0 2 2 1 2 0 0 1]
    
    [0 0 0 2 1 1 1 2 1 2 2 2 0 2]
    
    [0 0 0 0 1 1 1 0 1 1 1 0 1 0]
    
    [1 0 1 1 1 0 0 1 1 1 0 0 1 0]
    
    [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    
    Predicted Value: [1]
    
                  precision    recall  f1-score   support
    
               0       1.00      0.80      0.89         5
               1       0.90      1.00      0.95         9
    
        accuracy                           0.93        14
        
        macro avg       0.95      0.90      0.92        14
       
        weighted avg       0.94      0.93      0.93      14
