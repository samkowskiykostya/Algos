# coding=UTF-8

import pickle

import pandas as pd
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from test.MachineLearning.plots import *


if __name__ == '__main__':
    #Pipes usage
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    X = df.loc[:,2:].values; y = df.loc[:,1].values
    y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    pipe = Pipeline([('scl', StandardScaler()),
                     ('pca', PCA(n_components=2)),
                     ('clf', LogisticRegression(C=0.1))])
    pipe.fit(X_train, y_train)
    scores = cross_val_score(estimator=pipe, X=X_train, y=y_train, cv=10, n_jobs=-1)
    print(('CV accuracy scores: %s' % scores))
    print(('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))))
    print(('Test Accuracy: %.3f' % pipe.score(X_test, y_test)))

    #Learning and validation curves
    f = plt.figure()
    f.add_subplot(121); plot_learning_curves(pipe, X_train, y_train)
    f.add_subplot(122); plot_validation_curves(pipe, X_train, y_train)

    #Brute force search best params for learner
    pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
    param_range=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
                 {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
    #Can be used custom scorer
    make_scorer(score_func=precision_score, pos_label=1, greater_is_better=True, average='micro')
    gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print(('Best possible score: %s' % gs.best_score_))
    print(('Best possible SVM params: %s' % gs.best_params_))

    #Print confusion matrix
    y_pred = gs.predict(X_test)
    print((confusion_matrix(y_test, y_pred)))

    #Evaluate solution with crossval (no need to fit)
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print(('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))))

    #Save/restore model
    # pickle.dump(gs.best_estimator_, open('dump.txt','w'))
    print((pickle.load(open('dump.txt')).predict(X_test[0:2,:])))

    """
    presicion - TP/(TP+FP) - сколько из предсказанных тру оказались верны
    recall - TP/(TP+FN) - сколько из реально тру я угадал
    """
    print(('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred)))
    print(('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred)))
    print(('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred)))

    bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=None),
                           n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True,
                           bootstrap_features=False, n_jobs=-1, random_state=1)
    bag = bag.fit(X_train, y_train)
    y_train_pred = bag.predict(X_train)
    y_test_pred = bag.predict(X_test)
    print(('Bagging train/test accuracies %.3f/%.3f' % (accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred))))
    plt.show()


