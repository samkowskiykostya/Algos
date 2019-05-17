from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    X_train2 = X_train[:, [4, 14]]
    pipe_lr = Pipeline([
        ('slc', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('clf', LogisticRegression(penalty='l2', random_state=1, C=100))
    ])
    param = {
        'pca__n_components': [2, 4],
        'clf__C': [1, 10, 100],
        'clf__penalty': ['l1', 'l2']
             }
    grid = GridSearchCV(pipe_lr, param, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)
    print('Best estimator score: {}; {}'.format(grid.best_estimator_.score(X_test, y_test), grid.best_estimator_))

    pipe_lr.fit(X_train, y_train)
    print('Random estimator score: {}'.format(pipe_lr.score(X_test, y_test)))



def drawAUC():
    cv = list(StratifiedKFold(n_splits=6, random_state=2).split(X_train, y_train))
    f = plt.figure(figsize=(7, 5))
    mean_tpr=0
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train, test) in enumerate(cv):
        pbs = pipe_lr.fit(X_train2[train], y_train[train]).\
            predict_proba(X_train2[test])
        fpr, tpr, thresh = roc_curve(y_train[test], pbs[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold {}(area = {})'.format(i+1, roc_auc))
    plt.plot([0,1], [0,1], linestyle='--',label='random guessing')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()