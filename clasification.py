import  pandas as pd
import My_ML_Lib as mml
import pickle

if __name__ == '__main__':
    df = pd.read_csv('data_for_clasification.csv')
    df = df.sort_values(by=['baby_weight']).head(600)
    print(df['baby_weight'])
    X = df.drop('baby_weight', 1)
    Y = df['baby_weight']

    # Train our model/
    # using grid search method for select a proper hiperperamiter
    #
    # param_grid = {
    #     'C': [10,15,20,25,30,35,40],
    #
    # }
    # from sklearn.svm import LinearSVC
    # from sklearn.model_selection import GridSearchCV
    # lin = LinearSVC(loss='hing/e')
    # grid_search = GridSearchCV(estimator=lin, param_grid=param_grid,
    #                            cv=5, n_jobs=-1, verbose=2)
    # grid_search.fit(X, Y)
    # print('Final Loss', grid_search.best_params_, "Final accuracy :", grid_search.best_score_ * 100, "%")

    from sklearn.model_selection import train_test_split
    train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=.20)

    from sklearn.svm import LinearSVC
    lin=LinearSVC(loss='hinge',C=10)
    lin.fit(train_x,train_y)



    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score

    results = confusion_matrix(test_y, lin.predict(test_x))

    acc=accuracy_score(test_y, lin.predict(test_x))
    print(acc )


    import seaborn as sns
    import matplotlib.pyplot as plt

    ax = plt.subplot()
    sns.heatmap(results, annot=True, ax=ax);  # annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels',color='Blue');
    ax.set_ylabel('True labels',color='Blue');
    ax.set_title('Confusion Matrix', color='Red');
    ax.xaxis.set_ticklabels(['perfect weight','not perfect weight']);
    ax.yaxis.set_ticklabels(['perfect weight','not perfect weight']);

    plt.savefig('confusion_matrix.png')
    plt.show()

    # Train our model/
    #using grid search method for select a proper hiperperamiter
    # param_grid = {
    #     'C': [10],
    #
    # }
    # from sklearn.svm import LinearSVC
    #
    # lin = LinearSVC(loss='hinge')
    # grid_search = GridSearchCV(estimator=lin, param_grid=param_grid,
    #                            cv=5, n_jobs=-1, verbose=2)
    # grid_search.fit(X, Y)
    # print('Final Loss', grid_search.best_params_, "Final accuracy :", grid_search.best_score_ * 100, "%")

    from sklearn.metrics import accuracy_score

    # best_grid = grid_search.best_estimator_
    #
    # from sklearn.model_selection import learning_curve
    #
    # train_sizes, train_scores, valid_scores = learning_curve(
    #     best_grid, X, Y,train_sizes=[50, 80, 110], cv=5)
    #
    # print(train_scores,valid_scores)

    #
    # mml.save_model("model/svm {}.pkl".format(grid_search.best_score_ * 100), best_grid)

    # For input

    #
