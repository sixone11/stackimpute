#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys 
import os 
import argparse 
import pandas as pd 

class merge_data(object):

    """
    process data and train model
    """
    def create_parser(self,name):
        p = argparse.ArgumentParser(description = "remains to be modified")
        p.add_argument('-i', metavar = 'in-file',nargs="+",type=str,help = "input file")
        p.add_argument('-o', metavar = 'out-file', type = str, help = "outfile")
        p.add_argument('-m', metavar = 'merge-col',nargs="+",type=int,default=[0,1,2],help = "merge the file by col please input it one by one")
        return p

    def main(self,name,opts):
        tlist = []
        for i in range(len(opts.i)):
            temp = pd.read_csv(str(opts.i[i]),sep="\t",header=None)
            tlist.append(temp)
        out = pd.merge(tlist[0], tlist[1], how ='inner',suffixes=["_1","_2"],on=opts.m)
        # out = pd.merge(tlist[0], tlist[1], how ='outer',suffixes=["_1","_2"],on=opts.m)
        if len(opts.i)>2:
            for i in range(2,len(opts.i)):
                if i % 2 == 0:
                    suffix=[]
                    suffix.append("_" + str(i+1))
                    suffix.append("_" + str(i+2))
                #out = pd.merge(out,tlist[i], how='inner',suffixes=suffix,on=opts.m)
                out = pd.merge(out,tlist[i], how='outer',suffixes=suffix,on=opts.m)
        #out.to_csv(opts.o, index=False,sep="\t",encoding='gbk',float_format='%.0f',header=None)
        keep = (out.shape[1]-len(opts.m))/2 + len(opts.m)
        out = out.dropna(thresh=keep)
        out.drop([1],axis=1,inplace=True)
        out.to_csv(opts.o, index=False,sep="\t",encoding='gbk',na_rep='NaN',header=None)


    def run(self, args):
        self.name = os.path.basename(args[0])
        parser = self.create_parser(self.name)
        self.opts = parser.parse_args(args[1:])
        self.main(self.name, self.opts)
        

if __name__ == '__main__':
    merge_data().run(sys.argv)


