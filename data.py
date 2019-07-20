import pandas as pd
file=open('babies23.txt')

x=0
for i in file:

    col=i.split()
    break


x=0
li=[]
for i in file:

    if x!=0:
        li.append(i.split())

    x+=1
    #break



df = pd.DataFrame(li,columns=col)
df.to_csv('data.csv',index=False)
