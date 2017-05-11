import argparse
from collections import namedtuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from data import Data, char_featurizer, metadata_featurizer
from sklearn.externals import joblib
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten
from keras.layers import Conv1D, MaxPool1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, load_model
from sklearn.pipeline import make_pipeline
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier

from tabulate import tabulate

Split = namedtuple('Split', ['X_train', 'X_test', 'y_train', 'y_test'])


def create_cnn(timesteps, n):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(timesteps, n)),
        Conv1D(32, 3, activation='relu'),
        MaxPool1D(2),
        Dropout(0.25),

        Conv1D(32, 3, activation='relu'),
        Conv1D(32, 3, activation='relu'),
        MaxPool1D(2),
        Dropout(0.25),

        Flatten(),
        Dense(1),
        Activation('sigmoid')
    ])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


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


def save_pipeline(clf, data, path):
        nnet = clf.model
        nnet.save('nnet.h5')
        clf.model = None
        joblib.dump((clf, data), path)
        clf.model = nnet


def load_pipeline(path):
    model, data = joblib.load(path)
    model.model = load_model('nnet.h5')

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
    parser.add_argument('--network', '-n', choices=['nn', 'rnn', 'cnn'],
                        default='nn', help='the type of neural network to use')
    parser.add_argument('--load_from_disk', '-l', action='store_true',
                        help='load previously trained model from disk')

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


def get_model_and_data(args):
    data = Data(args.folder, args.pattern)

    if args.network == 'nn':
        featurizer = metadata_featurizer
    else:
        featurizer = char_featurizer

    X, y = data.sliding_window(2, featurizer)

    if args.network == 'nn':
        X = np.array(X)
    else:
        X = pad_sequences(X)

    if args.network == 'cnn':
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    split = Split(*train_test_split(X, y, test_size=args.test_size))
    X_train, y_train = downsample(split.X_train, split.y_train)
    X_test, y_test = downsample(split.X_test, split.y_test)

    if args.network == 'rnn':
        X_train = X_train[1:5000, :]
        y_train = y_train[1:5000]
        X_test = X_test[1:1000, :]
        y_test = y_test[1:1000]

    split = Split(X_train, X_test, y_train, y_test)

    print('{} training samples, {} testing samples'.format(
        split.X_train.shape[0], split.X_test.shape[0]))
    print('Number of features: {}'.format(split.X_train.shape[1]))

    if args.network == 'nn':
        clf = KerasClassifier(create_neuralnet, k=X.shape[1], dropout=args.dropout,
                              epochs=args.epochs, batch_size=32)
    elif args.network == 'rnn':
        clf = KerasClassifier(create_rnn, timesteps=X.shape[1], n=np.amax(X)+1,
                              epochs=args.epochs, batch_size=32)
    else:
        clf = KerasClassifier(create_cnn, timesteps=X.shape[1], n=X.shape[2],
                              epochs=args.epochs, batch_size=32)

    return clf, split


def main():
    args = get_args()

    if args.load_from_disk:
        clf, split = load_pipeline('model.pkl')
    else:
        clf, split = get_model_and_data(args)
        clf.fit(split.X_train, split.y_train)

    predictions = clf.predict(split.X_test)

    table = []
    table.append(['Accuracy', accuracy_score(split.y_test, predictions)])
    table.append(['f1', f1_score(split.y_test, predictions)])
    table.append(['Speech recall', recall_score(split.y_test, predictions)])
    table.append(['Speech precision', precision_score(split.y_test, predictions)])

    print()
    print(tabulate(table))

    if not args.load_from_disk:
        save_pipeline(clf, split, 'model.pkl')


if __name__ == '__main__':
    main()
