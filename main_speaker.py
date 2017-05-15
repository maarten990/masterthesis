import argparse

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from data import Data
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

from tabulate import tabulate
from models import (create_cnn, create_rnn, load_pipeline,
                    save_pipeline, Split)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder containing the training data')
    parser.add_argument('parsed_folder', help='folder containing the parsed data')
    parser.add_argument('-t', '--test_size', type=float, default=0.25,
                        help='ratio of testing date')
    parser.add_argument('-p', '--pattern', default='*.xml',
                        help='pattern to match in the training folder')
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='number of epochs to train for')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='the dropout ratio (between 0 and 1)')
    parser.add_argument('--network', '-n', choices=['rnn', 'cnn'],
                        default='rnn', help='the type of neural network to use')
    parser.add_argument('--load_from_disk', '-l', action='store_true',
                        help='load previously trained model from disk')
    parser.add_argument('--steps', type=int, default=5,
                        help='number of timesteps to use per sequence')

    return parser.parse_args()


def get_model_and_data(args):
    data = Data(args.folder, args.parsed_folder, args.pattern)

    X, y = data.speaker_timeseries(500, args.steps)
    y = to_categorical(y)

    split = Split(*train_test_split(X, y, test_size=args.test_size))

    print('{} training samples, {} testing samples'.format(
        split.X_train.shape[0], split.X_test.shape[0]))
    print('Number of features: {}'.format(split.X_train.shape[1]))

    if args.network == 'rnn':
        clf = KerasClassifier(create_rnn, timesteps=X.shape[1], n=np.amax(X) + 1,
                              n_outputs=y.shape[1], activation='softmax',
                              epochs=args.epochs, batch_size=32)
    else:
        clf = KerasClassifier(create_cnn, timesteps=X.shape[1], n=np.amax(X) + 1,
                              n_outputs=y.shape[1], activation='softmax',
                              epochs=args.epochs, batch_size=32)

    return clf, split


def extract_speaker(model, sentence, timesteps):
    seq = '0' * (timesteps - 1) + sentence
    X = []
    for i in range(0, len(seq) - timesteps):
        X.append(seq[i:(i + timesteps)])

    X = np.array([[ord(ch) for ch in seq] for seq in X])
    out = ''.join(chr(i) for i in model.predict(X))

    return out.replace('0', '')


def main():
    args = get_args()
    name = f'speaker_{args.network}'

    if args.load_from_disk:
        clf, split = load_pipeline(name)
    else:
        clf, split = get_model_and_data(args)
        clf.fit(split.X_train, split.y_train)

    if not args.load_from_disk:
        save_pipeline(clf, split, name)

    tests = ['Macaroni Sklepi, Bundesminister fur Winkels',
             'Marimba Wokkels (Die Linke)']
    print('\n'.join([extract_speaker(clf, t, args.steps) for t in tests]))

    predictions = clf.predict(split.X_test)

    table = []
    table.append(['Accuracy', accuracy_score(np.argmax(split.y_test, axis=1),
                                             predictions)])

    print()
    print(tabulate(table))

    import IPython
    IPython.embed()


if __name__ == '__main__':
    main()
