from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from practical.util import plotLearning

num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

m = Sequential([
    Conv2D(32, (3,3), padding='valid', input_shape=x_train.shape[1:], activation='relu'),
    Conv2D(32, (3,3), activation='relu'),
    AveragePooling2D(pool_size=(2,2)),
    # Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    # Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    # Dropout(0.5),
    Dense(num_classes, activation='softmax'),
])

m.compile(loss='categorical_crossentropy',
          optimizer='nadam',
          metrics=['accuracy'])

hist = m.fit(x_train, y_train,
      batch_size=32,
      epochs=200,
      validation_data=(x_test, y_test),
      shuffle=True,
      callbacks=[EarlyStopping(monitor='loss', patience=1, min_delta=0.0001)]
            )
m.save('cifarModel3.h5')

evaluation = m.evaluate(x_test, y_test)
print('Accuracy: {}'.format(evaluation))

plotLearning(hist)