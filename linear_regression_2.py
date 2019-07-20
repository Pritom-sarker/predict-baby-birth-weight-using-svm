import My_ML_Lib as my
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('data_for_clasification.csv' )

X=df.drop('baby_weight',1)
Y=df['baby_weight']
corel=df.corr()
#print(corel['wt'].sort_values())
score=[]
if __name__=='__main__':

    from sklearn.linear_model import SGDRegressor
    itrr=[]
    for itr in range(1,100000,25):
        sgd_reg = SGDRegressor(max_iter=itr, tol=-np.infty,
                               penalty=None, eta0=0.1, random_state=42
                               )

        from sklearn.model_selection import cross_val_score

        forest_scores = cross_val_score(sgd_reg, X, Y, cv=3,
                                        scoring="neg_mean_squared_error")
        forest_rmse_scores = np.sqrt(-forest_scores)
        print(forest_rmse_scores.mean())

        score.append(forest_rmse_scores.mean())
        itrr.append(itr)
        plt.scatter(itrr, score, alpha=0.5)
        plt.title('Cross validation score')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.savefig('score.jpg')

    my.save_model('finalized_model_using_regression_1.pkl', sgd_reg)

