#!/data/sixone/software/imputation/envs/stackimpute/bin/python3.8
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import defaultdict


'''
    Transfer the continuous features into binary features.

'''

#split by class number/pencentile 
def transfor_by_pencentile(df,col,class_number):
    arra=df[col].values
    separate=int(100/class_number)
    limitation=[np.percentile(arra,j) for j in [i for i in range(separate,100,separate)]]
    limitation=[-np.inf]+limitation+[np.inf]
    df[col]=pd.cut(df[col], bins=limitation)


def transfor_userdefined(df,col,limitation):
    arra=df[col].values
    df[col]=pd.cut(arra, bins=limitation)
    



#beta_up and down
def up_down_sample(df,sample_number):
    diff_chr=[np.nan for i in range(1,23)]
    for i in range(0,22):
        j=i+1
        diff_chr[i]=df[df[0]=="chr%s" %j]
    k450_up_chr=[np.nan for i in range(1,23)]
    k450_down_chr=[np.nan for i in range(1,23)]
    for i in range(1,23):
        i=i-1
        k450=diff_chr[i][sample_number].values
        k450_up_chr[i]=[np.nan for i in range(0,diff_chr[i].shape[0])]
        k450_down_chr[i]=[np.nan for i in range(0,diff_chr[i].shape[0])]
        range_up=[i for i in range(1,diff_chr[i].shape[0])]
        for j in range_up:
            k450_up_chr[i][j]=k450[j-1]
        range_down=[i for i in range(0,diff_chr[i].shape[0]-1)]
        for j in range_down:
            k450_down_chr[i][j]=k450[j+1]	
    k450_up=[]
    k450_down=[]
    for i in range(0,22):
        k450_up=k450_up+k450_up_chr[i]
        k450_down=k450_down+k450_down_chr[i]
    #new attribute dist
    df["beta_up"]=k450_up
    df["beta_down"]=k450_down
    return df

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
        up_chr[i][0]=beta_pre[1]
        #fill the first row beta down(not accurate)
        if(np.isnan(split_list[i].iloc[1,7])):
            down_chr[i][0]=beta_pre[1]
        else:
            down_chr[i][0]=split_list[i].iloc[1,7]
        #fill the first row beta down( not accurate)
        if(np.isnan(split_list[i].iloc[len(down_chr[i])-2,7])):
            up_chr[i][len(down_chr[i])-1]=beta_pre[len(split_list[i])-2]
        else:
            up_chr[i][len(down_chr[i])-1]=split_list[i].iloc[len(down_chr[i])-2,7]
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
        #new attribute dist
    new["beta_up"]=up
    new["beta_down"]=down
    new.reset_index(drop=True, inplace=True)
    return new   

