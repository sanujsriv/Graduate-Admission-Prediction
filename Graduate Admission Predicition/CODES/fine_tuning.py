# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 02:26:17 2019

@author: User
"""
import joblib
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

dataset= pd.read_csv('DATA\graduateAdmissions\Admission_Predict.csv',na_values=['.'])

data = dataset.drop(["Chance of Admit"], axis =1)
labels = dataset["Chance of Admit"]    

my_model_loaded = joblib.load("models/RandomForest.pkl")

param_grid = [ 
        {"n_estimators": [3,10,30,50,60,61], "max_features" :[2,4,6,8]},        
        {"bootstrap": [False] , "n_estimators": [3,10], "max_features": [2,3,4]},      
        ]

grid_search = GridSearchCV (my_model_loaded,param_grid,cv =5, scoring = "neg_mean_squared_error")

label_train=[1 if yt>=0.75 else 0 for yt in labels]
grid_search.fit(data,label_train)
#print(grid_search.best_params_)
#print(grid_search.best_estimator_)

cvres = grid_search.cv_results_
for mean_score,params in zip(cvres["mean_test_score"],cvres["params"]):
    i =0
    #print(np.sqrt(-mean_score),params)

encoder = LabelEncoder()   
encoder.fit_transform(data) 
feature_imprtance = grid_search.best_estimator_.feature_importances_
cat_one_hot_attribute = list(encoder.classes_)
attributes = [8] + cat_one_hot_attribute
#print()
#print(sorted(zip(feature_imprtance,attributes),reverse = True))
#print("feature_imprtance",feature_imprtance)


