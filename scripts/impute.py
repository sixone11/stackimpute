#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys 
import argparse
import time
import os.path
import random

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from collections import defaultdict
 
from stackimpute.metrics import *
from stackimpute.utils import *
from stackimpute.feature_process import *
from stackimpute.model_structure import *

class impute(object):

    """
    process data and train model
    """
    def create_parser(self,name):
        p = argparse.ArgumentParser(description = "parameter for imputation")
        p.add_argument('-i', metavar = 'in-file', type = str, help = "input file that has missing value")
        p.add_argument('-f', metavar = 'feature-file', type = str, help = "other feature")
        p.add_argument('-o', metavar = 'out-file', type = str, help = "outfile")
        p.add_argument('-m',action='store_true', help = "the species,if true:mouse",default=False)
        p.add_argument('-s',action='store_true', help = "save model or not, default not",default=False)
        p.add_argument('-t', metavar = 'training-rate', type = float, help = "the training rate")
        return p

    def main(self,name,opts):
        df = pd.read_csv(opts.i, sep="\t",header=None)
        df[[1]] = df[[1]].astype(int)
        df_beta = df.select_dtypes(include=['float64'])
        if(max(df_beta.max())>1):
           df_beta=df_beta/100
           df = pd.concat([df.iloc[:,0:2],df_beta],axis=1)
        df_beta = df_beta.values
        mean_beta=np.nanmean(df_beta,axis=1)
        var_beta=np.nanvar(df_beta,axis=1)
        class_matrix=df_beta
        class_matrix[class_matrix>=0.5] =1
        class_matrix[class_matrix<0.5] =0
        class_pre = defaultdict(list)
        class_pre_distribution=defaultdict(list)
        for i in [i for i in range(0,class_matrix.shape[0])]:
            if all(np.isnan(class_matrix[i,].tolist())):
                class_pre_distribution[i] = np.nan
                 # class_pre[i] = np.nan
                class_pre[i] = 0
            else:
                class_pre_distribution[i]=np.nonzero(class_matrix[i,]==0)[0].shape[0]/(np.nonzero(class_matrix[i,]==1)[0].shape[0]+np.nonzero(class_matrix[i,]==0)[0].shape[0])
                if class_pre_distribution[i]>=0.5:
                    class_pre[i]=0
                else:
                    class_pre[i]=1
    
        class_pre=list(class_pre.values())
        class_pre_distribution=list(class_pre_distribution.values())


        beta_pre_info=pd.DataFrame({'class_pre':class_pre,'class_ore_distribution':class_pre_distribution, 'mean_beta':mean_beta,'var_beta':var_beta})

        sample_number=df_beta.shape[1]
        beta_up=[np.nan for i in range(0,df.shape[0])]
        beta_down=[np.nan for i in range(0,df.shape[0])]
        #group by chr
        if(opts.m):
            chromosome=20
        else:
            chromosome=23

        diff_chr=[np.nan for i in range(1,chromosome)]
        for i in range(0,chromosome-1):
            j=i+1
            diff_chr[i]=df[df[0]=="chr%s" %j]

        distance_up_chr=[np.nan for i in range(1,chromosome)]
        distance_down_chr=[np.nan for i in range(1,chromosome)]


        for i in range(1,chromosome):
            i=i-1
            distance=diff_chr[i][1].values
            distance_up_chr[i]=[np.nan for i in range(0,diff_chr[i].shape[0])]
            distance_down_chr[i]=[np.nan for i in range(0,diff_chr[i].shape[0])]
            range_up=[i for i in range(1,diff_chr[i].shape[0])]
            for j in range_up:
                distance_up_chr[i][j]=distance[j]-distance[j-1]
            range_down=[i for i in range(0,diff_chr[i].shape[0]-1)]
            for j in range_down:
                distance_down_chr[i][j]=distance[j+1]-distance[j]
            #regulate the special position
            #8000 is user-specified
            distance_up_chr[i][0]=8000
            distance_down_chr[i][diff_chr[i].shape[0]-1]=8000

        distance_up=[]
        distance_down=[]

        for i in range(0,chromosome-1):
            distance_up=distance_up+distance_up_chr[i]
            distance_down=distance_down+distance_down_chr[i]


        count=0
        beta_up=defaultdict(list)
        beta_down=defaultdict(list)

        for i in [i for i in range(df.shape[1]-sample_number,df.shape[1])]:
            col=[0,1]
            col.append(i)
            sample=df[col]
            out=up_down_sample(sample,i)
            beta_up[count]=out["beta_up"]
            beta_down[count]=out["beta_down"]
            count=count+1

        # sample_number=df_beta.shape[1]
        distance_up=pd.DataFrame(distance_up)
        distance_up=distance_up.rename(columns={0:'distance_up'})
        distance_down=pd.DataFrame(distance_down)
        distance_down=distance_down.rename(columns={0:'distance_down'})
        constant=pd.concat([df[[i for i in range(0,df.shape[1]-sample_number)]],distance_up,distance_down,beta_pre_info],axis=1)
        constant=constant.rename(columns={0:'chr',1:'position'})


        process_data=[]
        for i in [i for i in range(df.shape[1]-sample_number,df.shape[1])]:
            count=i-df.shape[1]+sample_number
            sample_index=pd.DataFrame(np.array([i-df.shape[1]+sample_number for j in range(0,len(constant))]))
            beta=df[i].to_frame(name="beta")
            temp_data=pd.concat([constant,beta,beta_up[i-df.shape[1]+sample_number],beta_down[i-df.shape[1]+sample_number],sample_index],axis=1)
            temp_data=fill_up_down(temp_data,"chr")
            process_data.append(temp_data)

        process_data=pd.concat(process_data,axis=0)
        process_data=process_data.rename(columns={0:'sample_index'})
        process_data['index']=[i for i in range(0,process_data.shape[0])]
        process_data.reset_index(drop=True, inplace=True)
        beta=process_data["beta"]

        if opts.f:
            other_feature=pd.read_csv(opts.f, sep="\t",header=0)
            process_data=pd.merge(process_data,other_feature,on=['chr','position'],how='inner',sort=False)

        process_data=process_data.sort_values(by="index",ascending=True)
        process_data=process_data.drop("index",axis=1)
        process_data.reset_index(drop=True, inplace=True)

        for i in ['distance_up','distance_down']:
            transfor_userdefined(process_data,i,[-np.inf,200,np.inf])
        
        if not len(process_data.sample_index.unique())-1:
            process_data.drop('sample_index',axis=1,inplace=True)
        process_data=process_data.drop(["chr","beta",'position'],axis=1)
        func=lambda x: 1 if x>=0.5 else 0
        process_data['class_up']=pd.DataFrame(process_data['beta_up']).applymap(func)
        process_data['class_down']=pd.DataFrame(process_data['beta_down']).applymap(func)
        data_pipe_level =pipe_level.fit_transform(process_data)
       	data_pipe_level_withbeta=pd.concat([data_pipe_level,beta],axis=1)
        all_class=defaultdict(list)
        data_pipe_level_withbeta.reset_index(drop=True, inplace=True)

        all_class=pd.DataFrame(data_pipe_level_withbeta["beta"]).applymap(func)
        all_class=all_class.rename(columns={'beta':'class'})

    ###########################################################################################################################################################
        data_pipe_level_withbeta=pd.concat([data_pipe_level_withbeta,all_class],axis=1)
        data_pipe_level_withbeta['index']=[i for i in range(0,data_pipe_level_withbeta.shape[0])]
        

        if 'sample_index' in process_data.columns:
            pass
        else:
            data_pipe_level_withbeta.drop(['mean_beta','var_beta','class_ore_distribution','class_pre_0','class_pre_1'],axis=1,inplace=True)
        
        train=data_pipe_level_withbeta[data_pipe_level_withbeta["beta"].isnull()==False]
        train_all=train
        if(opts.t):
            from sklearn.model_selection import train_test_split
            self.seed=random.randrange(1,1000000000)
            train, t, train_y, t_y = train_test_split(train,train,train_size=opts.t,test_size=min(0.1,1-opts.t),random_state=self.seed)
        
        train_class=train['class']
        train_beta=train["beta"]
        train_index=train['index']
        train_index.reset_index(drop=True, inplace=True)
        train.reset_index(drop=True, inplace=True)
        train.drop(['class','beta','index'],axis=1,inplace=True)
        train.fillna(train.mean(),inplace=True)

     
        train_all_class=train_all['class']
        train_all_beta=train_all["beta"]
        train_all_index=train_all['index']
        train_all_index.reset_index(drop=True, inplace=True)
        train_all.reset_index(drop=True, inplace=True)        
        all_train_beta=train_all_beta
        all_train_index=train_all_index
        train_all.drop(['class','beta','index'],axis=1,inplace=True)


        test=data_pipe_level_withbeta[data_pipe_level_withbeta["beta"].isnull()==True]
        test_index=test['index']
        test.drop(['class','beta','index'],axis=1,inplace=True)
        test.fillna(test.mean(),inplace=True)
        self.missing_rate = round(1-train.shape[0]/(train.shape[0] + test.shape[0]),5)

        train_si = SimpleImputer().fit_transform(train)
        trainbeta_si = SimpleImputer().fit_transform(np.array(train_beta).reshape(-1,1)).ravel()
        
        stack_model.fit(train_si,trainbeta_si)
        self.stack_model = stack_model
        if(opts.s):
            import joblib
            outdir = os.path.dirname(opts.o)
            model_name=os.path.basename(opts.o).split(".",1)[0]
            joblib.dump(stack_model, outdir + "/"+ model_name+".pkl")

        predict=stack_model.predict(test.values)
        test_index.reset_index(drop=True, inplace=True)
        test_save=pd.DataFrame({'beta':predict, 'index':test_index})
        all_train_beta.reset_index(drop=True, inplace=True)
        all_train_index.reset_index(drop=True, inplace=True)
        train_save=pd.DataFrame({'beta':all_train_beta, 'index':all_train_index})
        save=pd.concat([test_save,train_save],axis=0)
        save.sort_values(by="index",ascending=True,inplace=True)
        save.drop(['index'],axis=1,inplace=True)
        self.result=pd.DataFrame(save.values.reshape(df.shape[0],sample_number))
        func=lambda x: 1 if x>=0.5 else 0
        class_result=self.result 
        class_result=class_result.applymap(func)
        self.result=pd.concat([df.iloc[:,0:2],self.result],axis=1)
        self.class_result=pd.concat([df.iloc[:,0:2],class_result],axis=1)
        

    def run(self, args):
        self.name = os.path.basename(args[0])
        parser = self.create_parser(self.name)
        self.opts = parser.parse_args(args[1:])
        self.main(self.name, self.opts)
        self.result.to_csv(self.opts.o,sep='\t',mode='a',header=None,index=False)
        self.class_result.to_csv(self.opts.o + ".class",sep='\t',mode='a',header=None,index=False)
        

if __name__ == '__main__':
    impute().run(sys.argv)


