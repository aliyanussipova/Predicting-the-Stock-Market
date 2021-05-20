import pandas as pd
import numpy as np
from datetime import datetime 

sp_df = pd.read_csv('sphist.csv')
sp_df['Date']= pd.to_datetime(sp_df['Date'])
date_filter = sp_df['Date'] > datetime(year=2015, month=4, day=1)

df_before_2015 = sp_df[datetime].copy()
df_before_2015.sort_values('Date', acsending=True)

python predict.py

for col,row in sp_df.iterrows():
    for row in range(0, len(sp_df):
                     col['day_5'] = np.average(sp_df['Date']-10, sp_df['Date'] -1)
                     
                     
df_clean = sp_df.drop(sp_df[sp_df['Date'] > datetime(year=1951, month=1, day=2)]
                      
df_clean = df_clean.dropna(axis=0)

train = df_clean[df_clean['Date'] < datetime(year=2013, month=1, day=1)]
test = df_clean[df_clean['Date'] > datetime(year=2013, month=1, day=1)]
                      
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error                     
                      
linear_model = LinearRegression()
features = df_clean.columns.drop('Close','High','Low', 'Open', 'Volume','Adj Close', 'Date')
                      
target = 'Close'
                      
linear_model.fit(train[[features]], train[target])
predict_train = linear_model.predict(test[features])
mse_train = mean_squared_error(test[target],predict_train)
rmse = np.sqrt(mse_train)

                      
                      

                      

                      


                      
                      
                      


