import argparse
import os.path
from collections import namedtuple

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
from torch.autograd import Variable

from tabulate import tabulate
from tqdm import trange
from data import sliding_window, pad_sequences
from models import LSTMClassifier, CNNClassifier, NameClassifier

Datatuple = namedtuple('Datatuple', ['X_is_speech', 'X_speaker', 'y_is_speech', 'Y_speaker'])


def load_model(filename):
    if os.path.exists(filename):
        return torch.load(filename)
    else:
        return (None, None)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder containing the training data')
    parser.add_argument('-p', '--pattern', default='*.xml',
                        help='pattern to match in the training folder')
    parser.add_argument('-tp', '--testpattern', default='*.xml',
                        help='pattern to match in the training folder for testing data')
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='number of epochs to train for')
    parser.add_argument('-d', '--dropout', type=float, default=0.2,
                        help='the dropout ratio (between 0 and 1)')
    parser.add_argument('--network', '-n', choices=['rnn', 'cnn'],
                        default='rnn', help='the type of neural network to use')

    return parser.parse_args()


def get_data(args, max_len=None):
    X_train, y_is_speech, Y_speaker, vocab = sliding_window(args.folder, args.pattern, 2, 0.1)
    X_train = pad_sequences(X_train, max_len)
    Y_speaker = pad_sequences(Y_speaker, max_len)

    X_test, y_test, Y_test, _ = sliding_window(args.folder, args.testpattern, 2, 0.1, vocab=vocab)
    X_test = pad_sequences(X_test, X_train.shape[1])
    Y_test = pad_sequences(Y_test, Y_speaker.shape[1])

    # filter the speaker extraction data to only positive samples
    X_train_spkr = X_train[y_is_speech == 1, :]
    Y_speaker = Y_speaker[y_is_speech == 1, :]
    X_test_spkr = X_test[y_test == 1, :]
    Y_test = Y_test[y_test == 1, :]

    print('Speech classifier: {} training samples, {} testing samples'.format(
        X_train.shape[0], X_test.shape[0]))
    print('Speaker extraction: {} training samples, {} testing samples'.format(
        X_train_spkr.shape[0], X_test_spkr.shape[0]))
    print('Sequence length: {}'.format(X_train.shape[1]))

    return (Datatuple(X_train, X_train_spkr, y_is_speech, Y_speaker),
            Datatuple(X_test, X_test_spkr, y_test, Y_test), vocab)


def train(model, X_train, y_train, epochs=100, batch_size=32, optimizer=None):
    model.train()

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters())

    losses = []
    t = trange(epochs, desc='Training')
    for _ in t:
        epoch_loss = Variable(torch.zeros(1)).float()

        for i in range(0, X_train.shape[0], batch_size):
            X = Variable(torch.from_numpy(X_train[i:i+32, :])).long()
            y = Variable(torch.from_numpy(y_train[i:i+32])).float()

            y_pred = model(X)
            loss = model.loss(y_pred, y)
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = epoch_loss.data[0]
        losses.append(loss)
        loss_delta = losses[-1] - losses[-2] if len(losses) > 1 else 0
        t.set_postfix({'loss': loss,
                       'Δloss': loss_delta})

    model.eval()
    return losses, optimizer


def evaluate_clf(model, X, y, print_pos=False, vocab=None):
    model.eval()
    predictions = model(Variable(torch.from_numpy(X)).long())
    predictions = predictions.squeeze().data.numpy()
    predictions = np.where(predictions > 0.5, 1, 0)
    print()

    table = []
    table.append(['f1', f1_score(y, predictions)])
    table.append(['Speech recall', recall_score(y, predictions)])
    table.append(['Speech precision', precision_score(y, predictions)])

    print()
    print(tabulate(table))
    print('Number of positive predictions:', len(predictions[predictions > 0.5]))
    print('Number of positive samples:', len(y[y > 0.5]))

    # print the positive classifications
    if print_pos:
        print()
        print('Speeches:')
        positives = X[predictions > 0.5, :]
        for i in range(positives.shape[0]):
            indices = positives[i, :]
            line = ' '.join(vocab.idx_to_token[idx] for idx in indices
                            if idx in vocab.idx_to_token)
            print(line)


def evaluate_spkr(model, X, y, idx_to_token):
    model.eval()
    predictions = model(Variable(torch.from_numpy(X)).long())

    correct = 0
    for i in range(X.shape[0]):
        full_string = X[i, :]
        true = y[i, :]
        pred = predictions.data.numpy()[i, :]

        true_words = full_string[true > 0.5]
        pred_words = full_string[pred > 0.5]

        if np.all(true_words == pred_words):
            correct += 1

    print()
    print(f'Speaker accuracy: {correct / X.shape[0]}')


def main():
    args = get_args()

    train_data, test_data, vocab = get_data(args, 40)

    clf_path = f'pickle/clf_{args.network}.pkl'
    spkr_path = f'pickle/spkr.pkl'
    clf_model, clf_optim = load_model(clf_path)
    spkr_model, spkr_optim = load_model(spkr_path)

    if clf_model is None:
        if args.network == 'rnn':
            clf_model = LSTMClassifier(input_size=len(vocab.token_to_idx) + 1,
                                       embed_size=128, hidden_size=32,
                                       num_layers=1, dropout=args.dropout)
        else:
            clf_model = CNNClassifier(input_size=len(vocab.token_to_idx) + 1,
                                      seq_len=train_data.X_is_speech.shape[1],
                                      embed_size=128, num_filters=16,
                                      dropout=args.dropout)

    if spkr_model is None:
        spkr_model = NameClassifier(input_size=len(vocab.token_to_idx) + 1,
                                    seq_length=train_data.X_speaker.shape[1],
                                    embed_size=128,
                                    encoder_hidden=64,
                                    num_layers=1,
                                    dropout=args.dropout)

    _, clf_optim = train(clf_model, train_data.X_is_speech, train_data.y_is_speech,
                         args.epochs, optimizer=clf_optim)
    torch.save((clf_model, clf_optim), clf_path)
    evaluate_clf(clf_model, test_data.X_is_speech, test_data.y_is_speech)

    _, spkr_optim = train(spkr_model, train_data.X_speaker, train_data.Y_speaker,
                          int(args.epochs / 2), optimizer=spkr_optim)
    torch.save((spkr_model, spkr_optim), spkr_path)
    evaluate_spkr(spkr_model, test_data.X_speaker, test_data.Y_speaker, vocab.idx_to_token)


if __name__ == '__main__':
    main()
