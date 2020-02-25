#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np




df1=pd.read_excel(r"D:\......")


##append new column as target variable
for i in range(len(df1)):
    lst.append(1)
    


df1['target_var']=lst


from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsRegressor



X = df1.iloc[:,1:3]
y = df1.iloc[:,3]



# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2018)



knn = KNeighborsRegressor(n_neighbors=5)
knn.fit( X_train , y_train )


y_pred = (knn.predict(X_test))


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np


print( np.sqrt( mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


###pass parameters to predictive model
datapoint_predict=knn.predict([[x1,y1]])


print(datapoint_predict.astype(int))







