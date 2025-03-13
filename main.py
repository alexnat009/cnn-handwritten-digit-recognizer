import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import ImageGrab, Image
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os


def load_dataset():
    """loading dataset of pre-drawn digits from mnist dataset
    preprocessing data:"""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train[:, :, :, np.newaxis]
    X_test = X_test[:, :, :, np.newaxis]
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    return X_train, X_test, y_train, y_test


def normalize(train, test):
    return tf.keras.utils.normalize(train), tf.keras.utils.normalize(test)


def getModel():
    modelF = tf.keras.models.Sequential()
    modelF.add(layer=tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu',
                                            input_shape=(28, 28, 1)))
    modelF.add(layer=tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu'))
    modelF.add(layer=tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
    modelF.add(layer=tf.keras.layers.Dropout(rate=0.25))
    modelF.add(layer=tf.keras.layers.Flatten())
    modelF.add(layer=tf.keras.layers.Dense(units=128, activation='relu'))
    modelF.add(layer=tf.keras.layers.Dense(units=10, activation='softmax'))
    # modelF.add(layer=tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_uniform', activation='relu'))
    # modelF.add(layer=tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    modelF.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelF


def evaluate_model(dataX, dataY, n_fold=5):
    scores, histories = [], []
    kfold = KFold(n_splits=n_fold, shuffle=True, random_state=1)
    model = getModel()
    for train_ix, test_ix in kfold.split(dataX):
        print("inLoop")
        X_train, X_test, y_train, y_test = dataX[train_ix], dataX[test_ix], dataY[train_ix], dataY[test_ix]
        history = model.fit(X_train, y_train, epochs=8, batch_size=128, validation_data=(X_test, y_test))
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        print('> %.3f' % (acc * 100.0))
        scores.append(acc)
        histories.append(history)
    return scores, histories


def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()


def summarize_performance(scores):
    # print summary
    print(f'Accuracy: mean={np.mean(scores) * 100} '
          f'std={np.std(scores) * 100},'
          f' n={len(scores)}')
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()


def run_test_harness():
    # load dataset
    X_train, X_test, y_train, y_test = load_dataset()

    # prepare pixel data
    X_train, X_test = normalize(X_train, X_test)

    # evaluation model with kfold-cross-validation
    scores, histories = evaluate_model(X_train, y_train)

    # learning curves
    summarize_diagnostics(histories)

    # summarize estimated performance
    summarize_performance(scores)


def loadModel(exists: bool, filename: str):
    X_train, X_test, y_train, y_test = load_dataset()

    def accuracy(model):
        print(model.summary())
        _, acc = model.evaluate(X_test, y_test)
        print(f'Accuracy: {acc * 100.0}')

    if not exists:
        # define a model
        model = getModel()
        # fit a model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        # save model
        model.save('model/final_model.h5')
        accuracy(model)
    else:
        # get already fitted model
        model = tf.keras.models.load_model(filename)
        # evaluate loaded model
        # accuracy(model)

    return model


def load_image(img):
    img = tf.keras.utils.img_to_array(img)
    img = img[np.newaxis, :, :, np.newaxis].astype(np.float_) / 255.
    return img


X_train, X_test, y_train, y_test = load_dataset()

# predict the class
model = loadModel(exists=True, filename='model/final_model.h5')

for i in range(10):
    img = load_image(X_test[i])
    predict_value = model.predict(img)
    digit = np.argmax(predict_value)
    print(f'Prediction {digit}\t Real value {np.where(y_test[i] == 1.)[0][0]}')

# Dataset subplots
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

fig, axes = plt.subplots(10, 10, figsize=(8, 8), sharey='all', sharex='all')
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i], cmap='binary', interpolation='nearest')
plt.show()
