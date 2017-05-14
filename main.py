import argparse

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from data import Data, char_featurizer, metadata_featurizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier

from tabulate import tabulate
from models import (create_cnn, create_neuralnet, create_rnn, load_pipeline,
                    save_pipeline, Split)


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
        save_pipeline(clf, split, f'predict_{args.network}')


if __name__ == '__main__':
    main()
