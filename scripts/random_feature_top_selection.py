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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from stackimpute.metrics import *
from stackimpute.utils import *
from stackimpute.feature_process import *
from stackimpute.model_structure import *


class random_feature_top_selection(object):

    def stackmodel_predict(data_asarray): 
            data_asframe = pd.DataFrame(data_asarray, columns=["f"+str(i) for i in range(0,aa.shape[1])])
            return stack_model.predict(data_asframe)
    
    def create_parser(self,name):
        p = argparse.ArgumentParser(description = "remains to be modified")
        p.add_argument('-i', metavar = 'in-file', type = str, help = "input file that has missing value")
        p.add_argument('-f', metavar = 'feature-file', type = str, help = "other feature")
        p.add_argument('-r', metavar = 'missing-rate', type = float, help = "default missing rate of data")
        p.add_argument('-t', metavar = 'training-rate', type = float, help = "the training rate")
        p.add_argument('-p', metavar = 'top p% rank', type = float, help = "top p% features using to train model")
        p.add_argument('-o', metavar = 'out-file', type = str, help = "outfile",default="./")
        p.add_argument('-m',action='store_true', help = "the species,if true:mouse",default=False)
        p.add_argument('-s',action='store_true', help = "save model or not, default not",default=False)
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
        #TODO
        #sample_name or sample_number
        for i in [i for i in range(df.shape[1]-sample_number,df.shape[1])]:
            col=[0,1]
            col.append(i)
            sample=df[col]
            out=up_down_sample(sample,i)
            beta_up[count]=out["beta_up"]
            beta_down[count]=out["beta_down"]
            count=count+1

    
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

        process_data=pd.merge(process_data,constant,how='inner',on=['chr','position'])
        beta=process_data["beta"]
        process_data.drop(['chr','position','beta','distance_up_x','distance_down_x','mean_beta_x','class_ore_distribution_y','var_beta_y','class_pre_x', 'class_ore_distribution_x', 'var_beta_x','class_pre_y'],axis=1,inplace=True)
        process_data.rename(columns={'distance_up_y':'distance_up','distance_down_y':'distance_down','mean_beta_y':'mean_beta'},inplace=True)
      
        

        func=lambda x: 1 if x>=0.5 else 0
        process_data['class_up']=pd.DataFrame(process_data['beta_up']).applymap(func)
        process_data['class_down']=pd.DataFrame(process_data['beta_down']).applymap(func)
        
        if len(process_data.sample_index.unique()):
            process_data.drop('sample_index',axis=1,inplace=True)
        ##############################################################
        data_pipe_level_withbeta=pd.concat([process_data,beta],axis=1)
        all_class=defaultdict(list)
        data_pipe_level_withbeta.reset_index(drop=True, inplace=True)

        all_class=pd.DataFrame(data_pipe_level_withbeta["beta"]).applymap(func)
        all_class=all_class.rename(columns={'beta':'class'})

    ###########################################################################################################################################################
        data_pipe_level_withbeta=pd.concat([data_pipe_level_withbeta,all_class],axis=1)
        data_pipe_level_withbeta['index']=[i for i in range(0,data_pipe_level_withbeta.shape[0])]
        all_class_beta_index=data_pipe_level_withbeta[['class','beta','index']]
        data_pipe_level_withbeta.drop(['class','beta','index'],axis=1,inplace=True)
        # when there is only 1 sample plese drop below features
        if 'sample_index' in process_data.columns:
            pass
        else:
            data_pipe_level_withbeta.drop(['mean_beta'],axis=1,inplace=True)
        

        train = data_pipe_level_withbeta[all_class_beta_index['beta'].isnull()==False]
        all_class_beta_index = all_class_beta_index[all_class_beta_index['beta'].isnull()==False] 
        if(opts.t):
            self.seed = random.randrange(1,1000000000)
            train, t, train_y, t_y = train_test_split(train,all_class_beta_index,train_size=opts.t,test_size=1-opts.t,random_state=self.seed)
        
       

        train_class=train_y['class']
        train_beta=train_y["beta"]
        train_index=train_y['index']
        train_index.reset_index(drop=True, inplace=True)
        train.reset_index(drop=True, inplace=True)
        
        all_train_beta=train_beta
        all_train_index=train_index
        train.fillna(train.mean(),inplace=True)

        train.reset_index(drop=True, inplace=True)
        train_array = SimpleImputer().fit_transform(train)
        train_beta_array = SimpleImputer().fit_transform(np.array(all_train_beta).reshape(-1,1)).ravel()

        stack_model.fit(train_array,train_beta_array)
        self.stack_model=stack_model

        # shap part
        def stackmodel_predict(data_asarray):
            data_asframe = pd.DataFrame(data_asarray, columns=["f"+str(i) for i in range(0,train.shape[1])])
            return stack_model.predict(data_asframe)

        
        X_summary = shap.kmeans(train,10)
        shap_kernel_explainer = shap.KernelExplainer(stackmodel_predict,X_summary)
        size=10
        train_part=train.iloc[1:size]
        shap_values = shap_kernel_explainer.shap_values(train_part)
        # shap_values_single = shap_kernel_explainer.shap_values(train.iloc[0,:], nsamples=1000)
        # print("shap_values =",shap_values)
        # print("base value =",shap_kernel_explainer.expected_value)

        outdir = opts.o+"/feature_selection"
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # shap.force_plot(shap_kernel_explainer.expected_value, shap_values_single, train.iloc[0,:],show=False,matplotlib=True)
        # plt.savefig(outdir+"/force_plot_single.png")

        f = plt.figure()
        shap.summary_plot(shap_values,train_part, plot_type="bar")
        f.savefig(outdir+"/summary_plot_bar.png", bbox_inches='tight', dpi=600)

        f = plt.figure()
        shap.summary_plot(shap_values,train_part)
        f.savefig(outdir+"/summary_plot.png", bbox_inches='tight', dpi=600)

        # f = plt.figure()
        # shap.dependence_plot("beta_up", shap_values,train_part,show=False)
        # f.savefig(outdir+"/dependence_plot.png", bbox_inches='tight', dpi=600)

        # f=shap.dependence_plot("beta_up", shap_values,train_part,show=False)
        # shap.save_html(outdir+"dependence_plot.htm", f)

        f=shap.force_plot(shap_kernel_explainer.expected_value, shap_values,train_part,show=False)
        shap.save_html(outdir+"/force_plot.htm", f)
        # select feature
        X_importance = train_part
        shap_sum = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame([X_importance.columns.tolist(), shap_sum.tolist()]).T
        importance_df.columns = ['column_name', 'shap_importance']
        importance_df = importance_df.sort_values('shap_importance', ascending=False)
        # top rate(%)
        rate = opts.p
        nrow = int(importance_df.shape[0]*rate)
        importance_part = importance_df.iloc[0:nrow,:]
        importance_part_name = importance_part.column_name.tolist()
        later_feature = ['beta_up','beta_down','class_up','class_down','distance_up','distance_down']
        importance_part_name = ['chr','position'] + [item for item in importance_part_name if not item in later_feature]
        feature_df = pd.read_csv(opts.f, sep="\t",header=0)  
        file = outdir +"feature_top_"+str(opts.p)+".out"
        feature_df[importance_part_name].to_csv(file,sep='\t',index=False)
        
        
    def write_table(self,file):
        if os.path.exists(file):
            self.to_csv(file,sep='\t',mode='a',header=None,index=False)
        else: 
            self.to_csv(file,sep='\t',mode='a',index=False)
    pd.DataFrame.write_table = write_table  

    def run(self, args):
        self.name = os.path.basename(args[0])
        parser = self.create_parser(self.name)
        self.opts = parser.parse_args(args[1:])
        self.main(self.name, self.opts)




if __name__ == '__main__':
    random_feature_top_selection().run(sys.argv)
