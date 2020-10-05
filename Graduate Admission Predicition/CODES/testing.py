# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 02:26:17 2019

@author: User
"""
import joblib
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit


dataset= pd.read_csv('DATA\graduateAdmissions\Admission_Predict_test.csv',na_values=['.'])

data_test = dataset.copy()

xtest=data_test.drop(['Chance of Admit'],axis=1)
ytest=data_test['Chance of Admit']

ylabel_test=[1 if yl>=0.75 else 0 for yl in ytest]


my_model_loaded = joblib.load("models/RandomForest.pkl")

final_predictions = my_model_loaded.predict(xtest)
final_mse = sklearn.metrics.mean_squared_error(ylabel_test,final_predictions)
final_rmse = np.sqrt(final_mse)
print()
print("final_test_rmse",final_rmse)
print()


