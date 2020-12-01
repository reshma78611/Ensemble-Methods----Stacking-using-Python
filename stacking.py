# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:43:38 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Boosting decision trees for classification
data=pd.read_csv('C:/Users/HP/Desktop/python prgrmg/bagging_boosting_stacking_RF/pima-indians-diabetes.data.csv')
data.columns=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
array=data.values
X=array[:,0:8]
Y=array[:,8]

#splitting data using K-fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold=KFold(n_splits=10,random_state=7)

#building decision tree, SVM, logistic regression using stacking
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
#create sub models
estimators = []
model1 = LogisticRegression(max_iter=500)
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

model=VotingClassifier(estimators)
results=cross_val_score(model,X,Y,cv=kfold)
print(results.mean())#0.762
