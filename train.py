import argparse
import os.path
from collections import namedtuple

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from tabulate import tabulate
from torch.autograd import Variable
from tqdm import trange

from data import sliding_window, pad_sequences
from models import LSTMClassifier, CNNClassifier, NameClassifier

Datatuple = namedtuple('Datatuple', ['X_is_speech', 'X_speaker', 'y_is_speech', 'Y_speaker'])


def load_model(filename):
    if os.path.exists(filename):
        return torch.load(filename)
    else:
        return (None, None)


def write_losses(losses, basename):
    batch, epoch = losses

    with open('batch_' + basename, 'w') as f:
        f.write('\n'.join([str(l) for l in batch]))

    with open('epoch_' + basename, 'w') as f:
        f.write('\n'.join([str(l) for l in epoch]))


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


def get_data(args, buckets=[-1], spkr_pad=-1):
    X, y_is_speech, Y, vocab = sliding_window(args.folder, args.pattern, 2, 0.1)
    Xb, yb = pad_sequences(X, y_is_speech, buckets)
    X, Y = pad_sequences(X, Y, [spkr_pad])

    Xt, yt, Yt, _ = sliding_window(args.folder, args.testpattern, 2, 0.1, vocab=vocab)
    Xtb, ytb = pad_sequences(Xt, yt, buckets)
    Xt, Yt = pad_sequences(Xt, Yt, [spkr_pad])

    # filter the spkr extraction data to only positive samples
    X_spkr = [b[y == 1, :] for b, y in zip(X, [y_is_speech])]
    Y = [b[y == 1, :] for b, y in zip(Y, [y_is_speech])]
    Xt_spkr = [b[y == 1, :] for b, y in zip(Xt, [yt])]
    Yt = [b[y == 1, :] for b, y in zip(Yt, [yt])]

    for i, X in enumerate(Xb):
        print('Speech classifier bucket {}: {} training samples, {} testing samples, sequence length {}'.format(
            i, X.shape[0], Xtb[i].shape[0], X.shape[1]))

    for i, X in enumerate(X_spkr):
        print('Speaker extraction bucket {}: {} training samples, {} testing samples, sequence length {}'.format(
            i, X.shape[0], Xt_spkr[i].shape[0], X.shape[1]))

    return (Datatuple(Xb, X_spkr, yb, Y),
            Datatuple(Xtb, Xt_spkr, ytb, Yt), vocab)


def train(model, X_buckets, y_buckets, epochs=100, batch_size=32, optimizer=None):
    model.train()

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters())

    batch_losses = []
    epoch_losses = []
    t = trange(epochs, desc='Training')
    for _ in t:
        epoch_loss = Variable(torch.zeros(1)).float()

        for X_train, y_train in zip(X_buckets, y_buckets):
            for i in range(0, X_train.shape[0], batch_size):
                X = Variable(torch.from_numpy(X_train[i:i+32, :])).long()
                y = Variable(torch.from_numpy(y_train[i:i+32])).float()

                y_pred = model(X)
                loss = model.loss(y_pred, y)
                epoch_loss += loss
                batch_losses.append(loss.data[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss = epoch_loss.data[0]
        epoch_losses.append(loss)
        loss_delta = epoch_losses[-1] - epoch_losses[-2] if len(epoch_losses) > 1 else 0
        t.set_postfix({'loss': loss,
                       'Δloss': loss_delta})

    model.eval()
    return (batch_losses, epoch_losses), optimizer


def evaluate_clf(model, Xb, yb):
    """
    Evaluate the trained model.
    Xb, yb: bucketed lists of training and test data
    """
    model.eval()
    predictions = []
    true = []

    for X, y in zip(Xb, yb):
        pred = model(Variable(torch.from_numpy(X)).long())
        pred = pred.squeeze().data.numpy()
        pred = np.where(pred > 0.5, 1, 0)
        predictions.extend(pred)
        true.extend(y)

    table = []
    table.append(['f1', f1_score(true, predictions)])
    table.append(['Speech recall', recall_score(true, predictions)])
    table.append(['Speech precision', precision_score(true, predictions)])

    print()
    print(tabulate(table))


def evaluate_spkr(model, Xb, yb, idx_to_token):
    model.eval()

    correct = 0
    for X, y in zip(Xb, yb):
        predictions = model(Variable(torch.from_numpy(X)).long())

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

    if args.network == 'rnn':
        buckets = [5, 10, 15, 25, 40, -1]
    else:
        buckets = [40]

    train_data, test_data, vocab = get_data(args, buckets, spkr_pad=40)

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
                                      seq_len=train_data.X_is_speech[0].shape[1],
                                      embed_size=128, num_filters=16,
                                      dropout=args.dropout)

    if spkr_model is None:
        spkr_model = NameClassifier(input_size=len(vocab.token_to_idx) + 1,
                                    seq_length=train_data.X_speaker[0].shape[1],
                                    embed_size=128,
                                    encoder_hidden=64,
                                    num_layers=1,
                                    dropout=args.dropout)

    clf_losses, clf_optim = train(clf_model, train_data.X_is_speech, train_data.y_is_speech,
                                  args.epochs, optimizer=clf_optim)
    torch.save((clf_model, clf_optim), clf_path)
    evaluate_clf(clf_model, test_data.X_is_speech, test_data.y_is_speech)
    write_losses(clf_losses, 'clf_losses.txt')

    spkr_losses, spkr_optim = train(spkr_model, train_data.X_speaker, train_data.Y_speaker,
                                    args.epochs, optimizer=spkr_optim)
    torch.save((spkr_model, spkr_optim), spkr_path)
    evaluate_spkr(spkr_model, test_data.X_speaker, test_data.Y_speaker, vocab.idx_to_token)
    write_losses(spkr_losses, 'spkr_losses.txt')


if __name__ == '__main__':
    main()
