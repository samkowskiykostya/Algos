from practical.util import plot_decision_regions
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping

def runIris():
    df = pd.read_csv('../datasets/iris/iris.data', header=None)#.sample(frac=1)
    classNum = 2
    y = pd.get_dummies(df.iloc[0:100, 4]).as_matrix()
    X = df.iloc[0:100, [0, 2]].as_matrix()
    model=Sequential([
        Dense(10, input_dim=X.shape[1], activation='relu'),
        Dense(classNum, activation='softmax')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['binary_accuracy']
    )
    history = model.fit(
        X, y,
        batch_size=1,
        epochs=50,
        callbacks=[EarlyStopping(monitor='loss', patience=1, min_delta=0.01)]
    )
    plt.figure(0)
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['loss'])
    plt.show()
    plt.figure(1)
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plot_decision_regions(X, y, classifier=model.predict)

runIris()