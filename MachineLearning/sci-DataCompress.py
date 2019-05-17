import pandas as pd
from sklearn.model_selection import train_test_split;
from sklearn.datasets import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from test.MachineLearning.plots import *


f = plt.figure()

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train); X_test_std = sc.transform(X_test)
X_stack = np.vstack((X_train_std, X_test_std)); Y_stack = np.hstack((y_train, y_test))

#No dimensinality reduction
ax=f.add_subplot(221)
lr=LogisticRegression()
lr.fit(X_train_std[:,[0,1]],y_train)
plot_decision_regions(X=X_stack[:,[0,1]], y=Y_stack, classifier=lr)
plt.text(0.1, 0.9,'LR by 2 random features', transform=ax.transAxes)

#PCA
ax=f.add_subplot(222)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std); X_test_pca=pca.transform(X_test_std)
X_stack_pca = np.vstack((X_train_pca, X_test_pca));
lr1=LogisticRegression()
lr1.fit(X_train_pca,y_train)
plot_decision_regions(X=X_stack_pca, y=Y_stack, classifier=lr1)
plt.text(0.1, 0.9,'LR with PCA', transform=ax.transAxes)

#LDA
ax=f.add_subplot(223)
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train); X_test_lda=lda.transform(X_test_std)
X_stack_lda = np.vstack((X_train_lda, X_test_lda));
lr2=LogisticRegression()
lr2.fit(X_train_lda,y_train)
plot_decision_regions(X=X_stack_lda, y=Y_stack, classifier=lr2)
plt.text(0.1, 0.9,'LR with LDA', transform=ax.transAxes)

#RBF kernel PCA
ax=f.add_subplot(224)
X,y = make_circles(n_samples=1000, noise=0.1, factor=0.23)
X_pca = KernelPCA(n_components=2, kernel='rbf', gamma=15).fit_transform(X)
plt.scatter(X[y==0,0], X[y==0,1], color='green', marker='o', alpha=0.5)
plt.scatter(X[y==1,0], X[y==1,1], color='yellow', marker='o', alpha=0.5)
plt.scatter(X_pca[y==0,0], X_pca[y==0,1], color='red', marker='^', alpha=0.5)
plt.scatter(X_pca[y==1,0], X_pca[y==1,1], color='blue', marker='^', alpha=0.5)
plt.tight_layout()
plt.text(0.1, 0.9,'RBF Kernel PCA', transform=ax.transAxes)

plt.tight_layout()
plt.show()