#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.stats as stats
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor


#indf is the dataframe you want to split:chr is the col name of the chromosome col
def split_by_chr(indf,chr):
    out_list=[np.nan for i in range(0,22)]
    for i in range(0,22):
        j=i+1
        out_list[i]=indf[indf[chr]=="chr%s" %j]
    return out_list

def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)
def delete_first_last(process_data,chr):
    split_list=split_by_chr(process_data,chr)
    for i in range(1,23):
        i=i-1
        #delete the first and the last row
        split_list[i]=split_list[i].drop(split_list[i].index[0])
        split_list[i]=split_list[i].drop(split_list[i].index[len(split_list[i])-1])
    for i in range(0,22):
        if(i==0):
            new=split_list[0]
        else:
            new=pd.concat([new,split_list[i]],axis=0)
    new.reset_index(drop=True, inplace=True)
    return new   


def evaluation(model,test,real_result):
    pred=model.predict(test)
    minus=real_result-pred
    num=len(pred[minus!=0])
    error_rate=num/len(test)
    accuracy=1-error_rate
    return(accuracy)


#through this function we can fill the missing beta_up and beta_down with pre info 
def fill_up_down(process_data,chr):
    split_list=split_by_chr(process_data,chr)
    up_chr=[np.nan for i in range(0,22)]
    down_chr=[np.nan for i in range(0,22)]
    for i in range(1,23):
        i=i-1
        up_chr[i]=[np.nan for i in range(0,split_list[i].shape[0])]
        down_chr[i]=[np.nan for i in range(0,split_list[i].shape[0])]
        beta_up=split_list[i]["beta_up"].values
        beta_down=split_list[i]["beta_down"].values
        beta_pre=split_list[i]["mean_beta"].values
        range_all=[i for i in range(1,split_list[i].shape[0]-1)]
        for j in range_all:
            if(np.isnan(beta_up[j])):
                up_chr[i][j]=beta_pre[j-1]
            else:
                up_chr[i][j]=beta_up[j]
        for j in range_all:
            if(np.isnan(beta_down[j])):
                down_chr[i][j]=beta_pre[j+1]
            else:
                down_chr[i][j]=beta_down[j]
        #fill the first row beta down(not accurate)
        if len(up_chr[i])==0:
            pass 
        else:
            up_chr[i][0]=beta_pre[1]
            #fill the first row beta down(not accurate)
            if(np.isnan(split_list[i].iloc[1,8])):
                down_chr[i][0]=beta_pre[1]
            else:
                down_chr[i][0]=split_list[i].iloc[1,8]
            #fill the first row beta down( not accurate)
            if(np.isnan(split_list[i].iloc[len(down_chr[i])-2,8])):
                up_chr[i][len(down_chr[i])-1]=beta_pre[len(split_list[i])-2]
            else:
                up_chr[i][len(down_chr[i])-1]=split_list[i].iloc[len(down_chr[i])-2,8]
            #fill the last row beta down( not accurate)
            down_chr[i][len(down_chr[i])-1]=beta_pre[len(split_list[i])-2]
    up=[]
    down=[]
    for i in range(0,22):
        if(i==0):
            new=split_list[0]
        else:
            new=pd.concat([new,split_list[i]],axis=0)
        up=up+up_chr[i]
        down=down+down_chr[i]
        # new attribute dist
    new["beta_up"]=up
    new["beta_down"]=down
    new.reset_index(drop=True, inplace=True)
    return new    


def transfor_userdefined(df,col,limitation):
    arra=df[col].values
    df[col]=pd.cut(arra, bins=limitation)




    

class level_label(BaseEstimator, TransformerMixin):
   def __init__(self):
        pass
    
   def fit(self,X,y=None):
        return(self)
    
   def transform(self,X):
        lab=LabelEncoder()
        process_name=X.select_dtypes(exclude=['number']).columns.values.tolist()
        X[process_name]=X[process_name].astype('str')
        for col in process_name:
            X[col]=lab.fit_transform(X[col])
        return X

class level_onehot_continuous(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        onehot_encoder = OneHotEncoder(sparse=False)
        x=defaultdict(list)
        j=0
        continues_name=X.select_dtypes(include=['float']).columns.values.tolist()
        binary_name=X.select_dtypes(exclude=['float']).columns.values.tolist()
        for col in binary_name:
            x[j]=pd.get_dummies(X[col],prefix=col)
            j=j+1
        temp1=[np.nan for i in range(0,len(x))]
        for i in range(0,len(x)):
            temp1[i]=x[i]
        x=temp1
        y=pd.concat(x,axis=1)
        y=pd.concat([y,X[continues_name]],axis=1)
        return y



pipe_level = Pipeline([('level_label', level_label()),('level_onehot_continuous', level_onehot_continuous())])



XGB = XGBRegressor(n_estimators=150,n_jobs=5)
GBR = GradientBoostingRegressor(loss="ls")
Bay = BayesianRidge()
LR = LinearRegression(n_jobs=5)
Ridge = Ridge(alpha=60)





def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse
    
    


class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,mod,meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)
        
    def fit(self,X,y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))
        
        for i,model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X,y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index,i] = renew_model.predict(X[val_index])
        
        self.meta_model.fit(oof_train,y)
        return self
    
    def predict(self,X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) 
                                      for single_model in self.saved_model]) 
        return self.meta_model.predict(whole_test)
    
    def get_oof(self,X,y,test_X):
        oof = np.zeros((X.shape[0],len(self.mod)))
        test_single = np.zeros((test_X.shape[0],5))
        test_mean = np.zeros((test_X.shape[0],len(self.mod)))
        for i,model in enumerate(self.mod):
            for j, (train_index,val_index) in enumerate(self.kf.split(X,y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index],y[train_index])
                oof[val_index,i] = clone_model.predict(X[val_index])
                test_single[:,j] = clone_model.predict(test_X)
            test_mean[:,i] = test_single.mean(axis=1)
        return oof, test_mean


stack_model = stacking(mod=[XGB,GBR,Bay,LR,Ridge],meta_model=XGB)





