import pandas as pd
df=pd.read_csv('data.csv')
#print(df.head())

new=pd.DataFrame()

# drop ->
dropp=['pluralty','date','outcome','id','sex','marital'] #all sex=1
df=df.drop(dropp,1)
df.info()



# colum need to be changed
# #Gastation
#   Gastaion - 280< gas <320  -> 0 risk or 1 no risk

gestation=df['gestation'].values

for i in range(0,len(gestation)):
    if gestation[i]>=280 and gestation[i]<=320:

        gestation[i]=1

    else:
        gestation[i] = 0

new['gestation_days']=gestation





#race-> 0=white, 6=mex, 7=black, 8=asian, 9=mix
race=df['race'].values
print(set(race))

x=0
for i in range(0,len(race)):
    if  race[i] <=5:
        race[i]=0

    if race[i] ==99:
        race[i] =0

    if race[i] == 10:
        race[i] = 0

print(set(race))
new['race']=race
# education
# mom education, 0=(<8), 1=(8-<12), 2=12, 3=12+trade, 4=12+some college, 5=16, 7=trade (hs unclear), 9=unknown


test=df['ed'].values
new['education']=test

# age - early{0} <=18 : 19<perfect{1}<34 : late>34{2}

age=df['age'].values
for i in range(0,len(age)):
    if age[i]<18:
        age[i]=0
    elif age[i]>=18 and age[i]<=34:
        age[i] = 1
    elif age[i]>34:
        age[i]=2

new['mom_age']=age

# weight & Height

ht=df['ht'].values
wt=df['mwt'].values
weight=[]
for i in range(0,len(ht)):
    miter=0.0254*ht[i]
    kg=0.453592*wt[i]
    bmi=kg/(miter*miter)

    if bmi<18.5:
        ht[i]=0
    elif bmi>=18.5 and bmi<=24.9:
        ht[i] = 1
    elif bmi> 24.9:
        ht[i] = 2


new['mother_bmi']=ht

#For dad
# age - early{0} <=21 : 21<perfect{1}<39 : late>39{2}

d_age=df['dage'].values

for i in range(0,len(d_age)):
    if d_age[i]<21:
        d_age[i]=0
    elif d_age[i]>=21 and d_age[i]<=39:
        d_age[i] = 1
    elif d_age[i]>39:
        d_age[i]=2

new['dad_age']=d_age
# weight & Height

htt=df['dht'].values
wt=df['dwt'].values
d_weight=[]
for i in range(0,len(htt)):
    miter=0.0254*htt[i]
    kg=0.453592*wt[i]
    bmi=kg/(miter*miter)

    if bmi<18.5:
        wt [i]=0
    elif bmi>=18.5 and bmi<=24.9:
        wt[i] = 1
    elif bmi> 24.9:
        wt[i] = 2


new['dad_bmi']=wt

#mother smoke 1= no smoke || 0= smoke
smoke=df['smoke'].values
for i in range(0,len(smoke)):
    if smoke[i]==1:
        smoke[i]=1
    else:
        smoke[i]=0

new['smoke']=smoke

#baby weights on  -> ounce
#
out=df['wt'].values
outcome=[]


for i in range(0,len(out)):
    if out[i]>88 and out[i]<135:
        out[i]=1

    else:
        out[i] = 0
    x+=1



new['baby_weight']=out

new.to_csv('data_for_clasification.csv', index=False)

