import argparse
import sys

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from data import Data, char_featurizer, metadata_featurizer
from keras import optimizers
from keras.layers import Activation, Dense, Dropout
from keras.layers.core import Masking
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier

from IPython import embed

def create_rnn(timesteps, n):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(timesteps, n)),
        LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        Activation('relu'),
        Dense(1),
        Dropout(0.2),
        Activation('sigmoid')
    ])

    model.compile(optimizer=optimizers.RMSprop(lr=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('t', '--test_size', type='float', default=0.2)


def main():
    data = Data(sys.argv[1], pattern='1800*.xml')

    X, y = data.sliding_window(3, char_featurizer)
    X = pad_sequences(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if True:
        X_train = X_train[1:5000, :, :]
        y_train = y_train[1:5000]
        X_test = X_test[1:1000, :, :]
        y_test = y_test[1:1000]

    print('{} training samples, {} testing samples'.format(X_train.shape[0], X_test.shape[0]))
    print('Number of features: {}'.format(X_train.shape[1]))

    #clf = LinearSVC()
    #clf = MLPClassifier(hidden_layer_sizes=(100, 25), verbose=True)
    clf = KerasClassifier(create_rnn, timesteps=X.shape[1], n=X.shape[2],
                          epochs=5, batch_size=32)

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, predictions))
    print('Speech recall: ', recall_score(y_test, predictions))
    print('Speech precision: ', precision_score(y_test, predictions))


if __name__ == '__main__':
    main()
