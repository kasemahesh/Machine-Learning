import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt 
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing as pp

medical = pd.read_csv('insurance.csv')

sober_data =medical[medical['smoker'] == 'no']




ages = sober_data.age

# print(sober_data.charges)
def predict_charges(ages,w,b):
    return w * ages + b

pcharges = predict_charges(ages,210,400)

# print(sober_data)
# print(pcharges)

def rmse(targets,preds):
    rmseval = np.sqrt(np.mean(np.square(preds-targets)))
    return rmseval

# print(rmse(sober_data['charges'],pcharges))

model = LinearRegression()

model.fit(sober_data[['age']],sober_data.charges)

# pcharges = model.predict(sober_data[['age']])
# print(pcharges)
# print(model.coef_)
# print(model.intercept_)
smoker_codes = {'no':0,'yes':1}
medical['smoker_code'] = medical.smoker.map(smoker_codes)

print(medical['charges'].corr(medical['smoker_code']))
enc = pp.OneHotEncoder()










# fig = px.scatter(sober_data,x='age',y='charges',color='smoker')
# fig.add_trace(
#     go.Scatter(
#         x=sober_data['age'], 
#         y=pcharges, 
#         mode='lines', 
#         name='Prediction',
#         line=dict(color='black', width=3)  # Customise line style
#     )
# )

# fig.show()