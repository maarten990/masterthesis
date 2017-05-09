import argparse
import sys

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from data import Data, char_featurizer, metadata_featurizer
from sklearn.externals import joblib
from keras.layers import Activation, Dense, Dropout, Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, load_model
from sklearn.pipeline import make_pipeline
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier


def create_rnn(timesteps, n):
    model = Sequential([
        Embedding(input_dim=n, input_length=timesteps, output_dim=64,
                  mask_zero=True),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        Dense(1),
        Activation('sigmoid')
    ])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def create_neuralnet(k, dropout):
    """ Create a simple feedforward Keras neural net with k inputs """
    model = Sequential([
        Dense(100, input_dim=k, activation='relu'),
        Dropout(dropout),
        Dense(25, activation='relu'),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def create_model(k, epochs, dropout):
    """
    Return an sklearn pipeline.
    k: the number of features to select
    """

    model = KerasClassifier(create_neuralnet, k=k, dropout=dropout,
                            epochs=epochs, batch_size=32)

    return make_pipeline(model)


def save_pipeline(model, data, path):
    if 'kerasclassifier' in model.named_steps:
        nnet = model.named_steps['kerasclassifier'].model
        nnet.save('nnet.h5')
        model.named_steps['kerasclassifier'].model = None
        joblib.dump((model, data), path)
        model.named_steps['kerasclassifier'].model = nnet

    else:
        joblib.dump((model, data), path)


def load_pipeline(path, keras=False):
    model, data = joblib.load(path)

    if keras:
        model.named_steps['kerasclassifier'].model = load_model('nnet.h5')

    return model, data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder containing the training data')
    parser.add_argument('-t', '--test_size', type=float, default=0.25,
                        help='ratio of testing date')
    parser.add_argument('-p', '--pattern', default='*.xml',
                        help='pattern to match in the training folder')
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='number of epochs to train for')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='the dropout ratio (between 0 and 1)')
    parser.add_argument('--features', '-f', choices=['meta', 'char'],
                        default='meta', help='the features to use')

    return parser.parse_args()


def downsample(X, y):
    "Downsample to balance the labels."
    n_positive = X[y == 1].shape[0]
    pos_X = X[y == 0][:n_positive, :]
    pos_y = y[y == 0][:n_positive]
    neg_X = X[y == 1]
    neg_y = y[y == 1]

    X_out = np.concatenate((pos_X, neg_X), axis=0)
    y_out = np.concatenate((pos_y, neg_y), axis=0)
    permutation = np.random.permutation(len(y_out))

    return X_out[permutation, :], y_out[permutation]


def main():
    args = get_args()

    data = Data(args.folder, args.pattern)

    if args.features == 'meta':
        featurizer = metadata_featurizer
    else:
        featurizer = char_featurizer

    X, y = data.sliding_window(2, featurizer)

    if args.features == 'char':
        X = pad_sequences(X)
        # X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    else:
        X = np.array(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)
    X_train, y_train = downsample(X_train, y_train)

    if args.features == 'char':
        X_train = X_train[1:5000, :]
        y_train = y_train[1:5000]
        X_test = X_test[1:1000, :]
        y_test = y_test[1:1000]

    print('{} training samples, {} testing samples'.format(X_train.shape[0], X_test.shape[0]))
    print('Number of features: {}'.format(X_train.shape[1]))

    if args.features == 'meta':
        clf = KerasClassifier(create_neuralnet, k=X.shape[1], dropout=args.dropout,
                              epochs=args.epochs, batch_size=32)
    else:
        clf = KerasClassifier(create_rnn, timesteps=X.shape[1], n=np.amax(X)+1,
                              epochs=args.epochs, batch_size=32)

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    print()
    print('Accuracy: ', accuracy_score(y_test, predictions))
    print('Speech recall: ', recall_score(y_test, predictions))
    print('Speech precision: ', precision_score(y_test, predictions))


if __name__ == '__main__':
    main()
