# Program 07: Decision Tree Classification (ID3 Algorithm)

## Question 7
Implement a Decision Tree classifier using the ID3 algorithm to predict whether a person has diabetes.

---

## 📂 Dataset

File: `diabetes.csv`

Columns:

1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age
9. Outcome

Where:
- Outcome = 1 → Diabetes
- Outcome = 0 → No Diabetes

## Output
    Predicted Output:
    [0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 1 0 1 1 1 1 0 0 0 0 0 0 1 0 0 0 0 1 0
     1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
     0 0 0 0 1 0 0 0 0 1 1 0 0 1 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0
     0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
     0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
     0 0 0 1 0 0 0 0 0]
    
    ID3 Classification Report:
                  precision    recall  f1-score   support
    
               0       0.72      0.91      0.80       151
               1       0.66      0.34      0.45        80
    
        accuracy                           0.71       231
       macro avg       0.69      0.62      0.62       231
    weighted avg       0.70      0.71      0.68       231

![](ID3_output.png)
