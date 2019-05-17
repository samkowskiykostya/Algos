import pandas as pd
from test.MachineLearning.plots import *

def tanFunc(sy):
    return 1 if sy > 0 else -1

class Perceptron:
    def __init__(self,X,Y,lr=0.1):
        self.X=X; self.Y=Y; self.lr=lr
        for i,x in enumerate(self.X):
            self.X[i] = [1]+self.X[i]
        self.W=np.zeros(len(self.X[0]))
    def evaluate(self, x):
        return tanFunc(np.dot(x,self.W))
    def predict(self, X):
        return np.array([self.evaluate(a) for a in X])
    def train(self, epochs=10):
        self.errors=[]
        f = plt.figure(3)
        for e in range(epochs):
            error=0
            for i,sampl in enumerate(self.X):
                z = self.evaluate(sampl)
                if z!= self.Y[i]:
                    error+=1
                for j,w in enumerate(self.W):
                    self.W[j] += self.lr*(self.Y[i]-z)*self.X[i][j]
            self.errors.append(error)
            f.add_subplot(330 + e + 1)
            plt.text(7,2, self.W)
            plt.scatter(self.X[:50, 0], self.X[:50, 1], color='red', marker='o', label='setosa')
            plt.scatter(self.X[50:100, 0], self.X[50:100, 1], color='blue', marker='x', label='versicolor')
            plot_decision_regions(self.X, self.Y, classifier=self)

def runSimpleBitOperation():
    p = Perceptron([[0,0],[0,1],[1,0],[1,1]], [1, -1, 1, -1], lr=0.08)
    p.train()
    for i,y in enumerate(p.Y):
        z = p.evaluate(p.X[i])
        print(z)
        assert y==z
    print('pass')

def runIris():
    df = pd.read_csv('../../datasets/iris/iris.data', header=None)
    y = np.where(df.iloc[0:100, 4] == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    p=Perceptron(X,y)
    p.train(9)
    # plt.figure(1)
    # plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    # plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # plot_decision_regions(X, y, classifier=p)
    plt.figure(2)
    plt.plot(list(range(1,len(p.errors)+1)), p.errors, marker='o')
    plt.show()

# runSimpleBitOperation()
runIris()