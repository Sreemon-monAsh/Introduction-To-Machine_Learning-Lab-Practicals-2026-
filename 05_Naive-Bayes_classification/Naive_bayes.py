age = [
    'youth','youth','middle-aged','senior','senior','senior',
    'middle-aged','youth','youth','senior','youth',
    'middle-aged','middle-aged','senior'
]

income = [
    'high','high','high','medium','low','low','low',
    'medium','low','medium','medium','medium','high','medium'
]

student = [
    'No','No','No','No','Yes','Yes','Yes',
    'No','Yes','Yes','Yes','No','Yes','No'
]

credit_rating = [
    'fair','excellent','fair','fair','fair','excellent',
    'excellent','fair','fair','fair','excellent',
    'excellent','fair','excellent'
]

buys_computer = [
    'no','no','yes','yes','yes','no','yes',
    'no','yes','yes','yes','yes','yes','no'
]

from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

le = preprocessing.LabelEncoder()

age_encoded = le.fit_transform(age)
print(age_encoded)
income_encoded = le.fit_transform(income)
print(income_encoded)
student_encoded = le.fit_transform(student)
print(student_encoded)
credit_encoded = le.fit_transform(credit_rating)
print(credit_encoded)

label = le.fit_transform(buys_computer)
print(label)

features = list(zip(age_encoded, income_encoded,student_encoded, credit_encoded))
model = GaussianNB()

model.fit(features, label)

predicted = model.predict([[2, 2, 1, 1]])
print("Predicted Value:", predicted)
print(classification_report(label, model.predict(features)))
