import statsmodels.api as sm
import pandas as pd 
import matplotlib.pyplot as plt  

data = pd.read_csv('../data scraping/swimmers_female.csv')

data = data[data['Power_Index']!= -1]   
#data = data[data['Power_Index']!= 100]
#data = data[data['Events-FR'] > 0] 

for index, row in data.iterrows():
	ppe = int(row['freshman_PPE']) 
	ppe += .01
	row['freshman_PPE'] = str(ppe)

data.plot(x='Power_Index',y='freshman_PPE',style='o',alpha=.3).invert_xaxis()   
plt.title('ACC Swimming')
plt.xlabel('Power_Index')
plt.ylabel('Points per Event')
plt.show()

Y = data['freshman_PPE']
X = sm.add_constant(data['Power_Index'])

gamma_model = sm.GLM(Y, X, family = sm.families.Gamma())

gamma_results = gamma_model.fit()

print(gamma_results.summary())