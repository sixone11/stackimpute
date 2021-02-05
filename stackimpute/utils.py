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

