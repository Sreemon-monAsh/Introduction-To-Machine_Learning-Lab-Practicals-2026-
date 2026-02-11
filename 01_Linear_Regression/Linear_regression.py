import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

file_path = "stock.csv"
df = pd.read_csv(file_path)

X = df[['Interest_Rate']].to_numpy()

Y = df['Stock_Index_Price']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept:', regr.intercept_)
print('Coefficient:', regr.coef_)

New_Interest_Rate = 2.75
print('Predicted Stock Index Price:',
      regr.predict([[New_Interest_Rate]]))

plt.scatter(df['Interest_Rate'], df['Stock_Index_Price'])
plt.title('Stock Index Price Vs Interest Rate')
plt.xlabel('Interest Rate')
plt.ylabel('Stock Index Price')
plt.grid(True)
plt.show()
