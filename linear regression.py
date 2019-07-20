import My_ML_Lib as my
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('data_for_clasification.csv' )

X=df.drop('baby_weight',1)
Y=df['baby_weight']
corel=df.corr()
print(corel['baby_weight'].sort_values())



df=pd.read_csv('data_for_regression.csv' )

X=df.drop('wt',1)
Y=df['wt']
corel=df.corr()
print(corel['wt'].sort_values())



my.plot_for_linear_relation(X,Y,'wt','sow')
# score=[]
# if __name__=='__main__':
#
#     from sklearn.linear_model import SGDRegressor
#     itrr=[]
#
#
#     sgd_reg = SGDRegressor(max_iter=1e5, tol=-np.infty,
#                            penalty=None, eta0=0.1, random_state=42
#                            )
#     from sklearn.model_selection import learning_curve
#     from sklearn.svm import SVC
#
#     train_sizes, train_scores, valid_scores = learning_curve(
#         sgd_reg, X, Y, cv=5)
#     print(train_sizes, train_scores, valid_scores)
#     plt.plot(train_scores)
#     plt.show()
#
