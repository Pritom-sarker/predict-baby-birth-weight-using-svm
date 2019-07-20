import sklearn
import pandas as pd
import numpy as np

df=pd.read_csv('data.csv' )

dropp=['pluralty','date','outcome','id','sex','marital','time','number']
df=df.drop(dropp,1)

smoke=df['smoke'].values

for i in range(0,len(smoke)):
    if smoke[i]==1:
        smoke[i]=1
    else:
        smoke[i]=0

df=df.drop('smoke',1)

df['smoke']=smoke


df.to_csv('data_for_regression.csv',index=False)