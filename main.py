import  pandas as pd
import My_ML_Lib as mml
import pickle

if __name__ == '__main__':
    df = pd.read_csv('data_for_clasification.csv')

    df=df.sort_values(by=['baby_weight']).head(600)
    x=df['baby_weight'].values
    s=0
    for i in x:

        if i==1:
            s+=1
    print(s)