import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
# %matplotlib inline
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import utils

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)

x_train = x_train / 255
x_test = x_test / 255

plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i]])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer=keras.optimizers.SGD(), loss=('sparse_categorical_crossentropy'), metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=2000)


