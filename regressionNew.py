import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


data = pd.read_csv('../data scraping/swimmers_male.csv')

data = data[data['Power_Index']!= -1]
data = data[data['Power_Index']!= 100]
data = data[data['Events-FR'] > 0] 

data.plot(x='Power_Index',y='freshman_PPE',style='o',alpha=.3).invert_xaxis()
plt.title('ACC Swimming')
plt.xlabel('Power_Index')
plt.ylabel('Points per Event')
plt.show()

x = data['Power_Index'].values.reshape(-1,1)
y = data['freshman_PPE'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df.head(30))

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))