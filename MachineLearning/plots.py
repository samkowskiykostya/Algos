import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import learning_curve, validation_curve
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
def plotfunc(f, x):
    y=[]
    for xi in x:
        try:
            y.append(f(xi))
        except:
            y.append(None)
    plt.plot(x,y)
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot3Dfunc(f, x, y):
    z=[]
    for i in range(len(x)):
        for j in range(len(y)):
            try:
                z.append(f(x[i], y[j]))
            except:
                z.append(None)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = np.meshgrid(np.arange(len(x)), -np.arange(len(y)), z)
    ax.plot_surface(X, Y, Z, alpha=0.5, cstride=2, rstride=2)
    plt.tight_layout()
    plt.show()

def plot3Dscatter(vals, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vals[:,0], vals[:,1], vals[:,2])
    for i in range(len(labels)):
        ax.text(vals[i,0],vals[i,1],vals[i,2], labels[i])


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                alpha=1.0, linewidth=1, marker='o',
                s=55, label='test set')

def plot_learning_curves(pipe_lr, X_train, y_train):
    train_sizes, train_scores, test_scores =\
                    learning_curve(estimator=pipe_lr,
                    X=X_train,
                    y=y_train,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    cv=10,
                    n_jobs=1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')
    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.tight_layout()
    # plt.savefig('./figures/learning_curve.png', dpi=300)

def plot_validation_curves(pipe_lr, X_train, y_train, param_range=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]):
    train_scores, test_scores = validation_curve(
                    estimator=pipe_lr,
                    X=X_train,
                    y=y_train,
                    param_name='clf__C',
                    param_range=param_range,
                    cv=10)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')
    plt.plot(param_range, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')
    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.8, 1.0])
    plt.tight_layout()

# plot3Dfunc(lambda x,y: x + y, list(range(-10, 10)), list(range(-10,10)))