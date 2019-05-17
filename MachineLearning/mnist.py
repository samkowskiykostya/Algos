import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import np_utils

def load_mnist(path, which='train'):
    if which == 'train':
        labels_path = os.path.join(path, 'train-labels-idx1-ubyte')
        images_path = os.path.join(path, 'train-images-idx3-ubyte')
    elif which == 'test':
        labels_path = os.path.join(path, 't10k-labels-idx1-ubyte')
        images_path = os.path.join(path, 't10k-images-idx3-ubyte')
    else:
        raise AttributeError('`which` must be "train" or "test"')
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, n, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.title('true label: %d' % y[idx])
    plt.show()

X,y = load_mnist('../../datasets/mnist')
X_t,y_t = load_mnist('../../datasets/mnist', which='test')
y_c=np_utils.to_categorical(y)

model = Sequential()
model.add(Dense(input_dim=X.shape[1],output_dim=50,init='uniform',activation='tanh'))
model.add(Dense(input_dim=50, output_dim=50,init='uniform',activation='tanh'))
model.add(Dense(input_dim=50, output_dim=y_c.shape[1],init='uniform',activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, decay=1e-7, momentum=0.9))
model.fit(X,y_c,nb_epoch=50,batch_size=300,verbose=1,validation_split=0.1,show_accuracy=True)
y_t_p = model.predict_classes(X_t, verbose=0)
print(('Test accuracy: %.2f%%' % (float(np.sum(y_t == y_t_p, axis=0)) / X_t.shape[0] * 100)))