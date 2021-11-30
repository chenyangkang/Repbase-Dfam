# %%
import pandas as pd
import sys
import math

# %%
data_basic_ph=pd.read_csv("basic_pheno.txt",sep="\t",header=None)
data_all_ph=pd.read_csv("all.pheno",sep="\t")
data_basic_ph_out=data_basic_ph.copy()

# %%
# for index1,line in data_basic_ph.iterrows():
#     for index2,value in data_all_ph.iterrows():
#         if line[1]==value['IID']:
#             print("yes")
#             data_basic_ph_out[5][index1]=data_all_ph.loc[index2,sys.argv[1]]

# %%
data1=data_all_ph.loc[:,["IID",sys.argv[1]]]
data2=data_basic_ph_out
data2.columns=["FID","IID","paternal","maternal","sex","phenotype"]
del data2["phenotype"]
data_basic_ph_out=pd.merge(data2,data1,on="IID")
for index,line in data_basic_ph_out.iterrows():
    if math.isnan(line[sys.argv[1]]):
        data_basic_ph_out[sys.argv[1]][index]="NA"



# %%
data_basic_ph_out=data_basic_ph_out.loc[:,["FID","IID",sys.argv[1]]]

# %%
data_basic_ph_out.to_csv("basic_pheno.%s.txt"%(sys.argv[1]),index=None,sep="\t",header=None)



