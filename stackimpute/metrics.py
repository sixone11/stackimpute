#!/usr/bin/env python
# -*- coding: utf-8 -*-

# classification
import numpy as np
import pandas as pd 
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
def specifity(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]/\
    (confusion_matrix(y_true, y_pred)[1, 1]+\
        confusion_matrix(y_true, y_pred)[0, 1])

def sensitivity(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]/\
    (confusion_matrix(y_true, y_pred)[0, 0]+\
        confusion_matrix(y_true, y_pred)[1, 0])

def accuracy(y_true, y_pred): return (confusion_matrix(y_true, y_pred)[0, 0]+\
    confusion_matrix(y_true, y_pred)[1, 1])/\
        (confusion_matrix(y_true, y_pred)[1, 1]+\
            confusion_matrix(y_true, y_pred)[0, 1]+\
                confusion_matrix(y_true, y_pred)[1, 0]+\
                    confusion_matrix(y_true, y_pred)[0, 0])

def precision(y_true, y_pred): return(confusion_matrix(y_true, y_pred)[0, 0]/\
    (confusion_matrix(y_true, y_pred)[0, 0]+confusion_matrix(y_true, y_pred)[0, 1]))

def recall(y_true, y_pred): return(confusion_matrix(y_true, y_pred)[0, 0]/\
    (confusion_matrix(y_true, y_pred)[0, 0]+confusion_matrix(y_true, y_pred)[1, 0]))

def MCC(y_true, y_pred): 
    a1=confusion_matrix(y_true, y_pred)[0, 0]+confusion_matrix(y_true, y_pred)[1, 0]
    a2=confusion_matrix(y_true, y_pred)[0, 0]+confusion_matrix(y_true, y_pred)[0, 1]
    a3=confusion_matrix(y_true, y_pred)[1, 1]+confusion_matrix(y_true, y_pred)[0, 1]
    a4=confusion_matrix(y_true, y_pred)[1, 1]+confusion_matrix(y_true, y_pred)[1, 0]
    denominator= np.float64(a1)* np.float64(a2)* np.float64(a3)* np.float64(a4)
    denominator=np.sqrt(denominator)
    numerator=confusion_matrix(y_true, y_pred)[0, 0]*confusion_matrix(y_true, y_pred)[1, 1]-(confusion_matrix(y_true, y_pred)[0, 1]*confusion_matrix(y_true, y_pred)[1, 0])
    return numerator/denominator
           

def tp(y_true, y_pred):return confusion_matrix(y_true, y_pred)[0, 0]
def tn(y_true, y_pred):return confusion_matrix(y_true, y_pred)[1, 1]
def fp(y_true,y_pred):return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true,y_pred):return confusion_matrix(y_true, y_pred)[1, 0]

def metrics_all(y_true,y_pred,proba,index=0): 
    return pd.DataFrame({'tn':tn(y_true,y_pred),\
        'tp':tp(y_true,y_pred),\
            'fp':fp(y_true,y_pred),\
                'fn':fn(y_true,y_pred),\
                    'specifity':specifity(y_true,y_pred),\
                        'sensitivity':sensitivity(y_true,y_pred),\
                            'accuracy':accuracy(y_true,y_pred),\
                                'precision':precision(y_true,y_pred),\
                                    'recall':recall(y_true,y_pred),\
                                        'MCC':MCC(y_true,y_pred),\
                                            'AUC':roc_auc_score(y_true,proba)},\
                                    index=[index])

# As a dict mapping the scorer name to the scoring function
classification_scoring = \
    {'accuracy':make_scorer(accuracy) ,\
           'sensitivity':make_scorer(sensitivity),\
               'specifity':make_scorer(specifity),\
                   'MCC':make_scorer(MCC),\
                       'tp':make_scorer(tp),\
                           'tn':make_scorer(tn),\
                               'fp':make_scorer(fp),\
                                   'fn':make_scorer(fn)}
                

# Regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

regression_scoring = {'RMSE':mean_squared_error ,\
           'R2':r2_score}


import random
def cross_validation_self(clf,stack_model,data,data_y,data_beta,n):
    all_sites = data.shape[0]
    part_sites = int(all_sites/n)
    N = range(n)
    result=pd.DataFrame()
    for i in N:
        index = random.sample(range(0,all_sites),part_sites)
        train_data = data.iloc[index]
        train_data_y = data_y.iloc[index]
        train_data_pred = clf.predict(train_data)
        train_data_proba = pd.DataFrame(clf.predict_proba(train_data))[1]
        train_data=pd.concat([train_data,pd.get_dummies(train_data_y,prefix="class")],axis=1)
        for_train = Imputer().fit_transform(train_data)
        for_train_y = Imputer().fit_transform(np.array(data_beta).reshape(-1,1)).ravel()
        stack_model.fit(for_train,for_train_y)

        predict=stack_model.predict(test.values)

        temp = metrics_all(train_data_y,train_data_pred,train_data_proba,i)
        result = pd.concat([result,temp],axis=0)
    # all to test 
    data_pred = clf.predict(data)
    data_proba = pd.DataFrame(clf.predict_proba(data))[1]
    temp = metrics_all(data_y,data_pred,data_proba,n)
    result = pd.concat([result,temp],axis=0)
    return(result)








