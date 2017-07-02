import argparse
import os.path
from collections import namedtuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from data import sliding_window, pad_sequences
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tabulate import tabulate
from tqdm import trange

Datatuple = namedtuple('Datatuple', ['X', 'y_is_speech', 'Y_speaker'])


class CNNClassifier(nn.Module):
    def __init__(self, input_size, seq_len, embed_size, num_filters, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embed_size)
        self.pool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(embed_size, num_filters, 3)

        c2_size = self.pool(self.conv1(Variable(torch.zeros(32, embed_size, seq_len)))).size(2)
        self.conv2 = nn.Conv1d(c2_size, num_filters, 3)

        clf_size = self.pool(
            self.conv2(Variable(torch.zeros(32, c2_size, num_filters)))).size(2) * num_filters
        self.clf_h = nn.Linear(clf_size, int(clf_size / 2))
        self.clf_out = nn.Linear(int(clf_size / 2), 1)

    def forward(self, inputs):
        embedded = self.embedding(inputs)

        # permute from [batch, seq_len, input_size] to [batch, input_size, seq_len]
        embedded = embedded.permute(0, 2, 1)
        l1 = self.dropout(self.pool(self.conv1(embedded)))
        l2 = self.dropout(self.pool(self.conv2(l1.permute(0, 2, 1))))

        batch_size = inputs.size(0)
        clf_in = l2.view(batch_size, -1)
        h = self.dropout(F.sigmoid(self.clf_h(clf_in)))
        out = F.sigmoid(self.clf_out(h))

        return out


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers,
                           bidirectional=True, batch_first=True,
                           dropout=dropout)

        # the output size of the rnn is 2 * hidden_size because it's bidirectional
        self.clf_h = nn.Linear(hidden_size * 2, hidden_size)
        self.clf_out = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        # initialize the lstm hidden states
        hidden = self.init_hidden(inputs.size(0))
        cell = self.init_hidden(inputs.size(0))

        # run the LSTM over the full input sequence and take the average over
        # all the outputs
        embedded = self.embedding(inputs)
        outputs, _ = self.rnn(embedded, (hidden, cell))
        averaged = torch.mean(outputs, dim=1).squeeze()

        # sigmoid classification with 1 hidden layer in between
        hiddenlayer = self.dropout(F.sigmoid(self.clf_h(averaged)))
        out = F.sigmoid(self.clf_out(hiddenlayer))

        return out

    def init_hidden(self, batch_size):
        "Initialize a zero hidden state with the appropriate dimensions."
        hidden = Variable(torch.zeros(1, self.hidden_size))
        hidden = hidden.repeat(self.num_layers * 2, batch_size, 1)
        return hidden


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder containing the training data')
    parser.add_argument('-p', '--pattern', default='*.xml',
                        help='pattern to match in the training folder')
    parser.add_argument('-tp', '--testpattern', default='*.xml',
                        help='pattern to match in the training folder for testing data')
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='number of epochs to train for')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='the dropout ratio (between 0 and 1)')
    parser.add_argument('--network', '-n', choices=['rnn', 'cnn'],
                        default='rnn', help='the type of neural network to use')

    return parser.parse_args()


def get_data(args, max_len=None):
    X_train, y_is_speech, Y_speaker, vocab = sliding_window(args.folder, args.pattern, 2, 0.01)
    X_train = pad_sequences(X_train, max_len)
    Y_speaker = pad_sequences(Y_speaker, max_len)
    
    X_test, y_test, Y_test, _ = sliding_window(args.folder, args.testpattern, 2, 0.01, vocab=vocab)
    X_test = pad_sequences(X_test, X_train.shape[1])
    Y_test = pad_sequences(Y_test, Y_speaker.shape[1])

    print('{} training samples, {} testing samples'.format(
        X_train.shape[0], X_test.shape[0]))
    print('Sequence length: {}'.format(X_train.shape[1]))

    return Datatuple(X_train, y_is_speech, Y_speaker), Datatuple(X_test, y_test, Y_test), vocab


def train(model, X_train, y_train, epochs=100, batch_size=32):
    model.train()
    optimizer = torch.optim.RMSprop(model.parameters())

    losses = []
    t = trange(epochs, desc='Training')
    for _ in t:
        epoch_loss = Variable(torch.zeros(1)).float()

        for i in range(0, X_train.shape[0], batch_size):
            X = Variable(torch.from_numpy(X_train[i:i+32, :])).long()
            y = Variable(torch.from_numpy(y_train[i:i+32])).float()

            y_pred = model(X)
            loss = F.binary_cross_entropy(y_pred, y)
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = epoch_loss.data[0]
        losses.append(loss)
        loss_delta = losses[-1] - losses[-2] if len(losses) > 1 else 0
        t.set_postfix({'loss': loss,
                       'Δloss': loss_delta})

    model.train(False)
    return losses


def main():
    args = get_args()

    train_data, test_data, vocab = get_data(args, 40)

    if args.network == 'rnn':
        clf_model = LSTMClassifier(input_size=len(vocab.token_to_idx) + 1,
                                       embed_size=128, hidden_size=32,
                                       num_layers=1, dropout=args.dropout)
    else:
        clf_model = CNNClassifier(input_size=len(vocab.token_to_idx) + 1,
                                      seq_len=train_data.X.shape[1],
                                      embed_size=128, num_filters=16,
                                      dropout=args.dropout)

    clf_path = f'pickle/clf_{args.network}.pkl'
    if os.path.exists(clf_path):
        try:
            clf_model.load_state_dict(torch.load(clf_path))
        except (RuntimeError, KeyError) as e:
            print(f'Could not load previous model: {e}')

    train(clf_model, train_data.X, train_data.y_is_speech, args.epochs)
    torch.save(clf_model.state_dict(), clf_path)

    predictions = clf_model(Variable(torch.from_numpy(test_data.X)).long())
    predictions = predictions.squeeze().data.numpy()
    predictions = np.where(predictions > 0.5, 1, 0)
    print()

    table = []
    table.append(['Accuracy', accuracy_score(test_data.y_is_speech, predictions)])
    table.append(['f1', f1_score(test_data.y_is_speech, predictions)])
    table.append(['Speech recall', recall_score(test_data.y_is_speech, predictions)])
    table.append(['Speech precision', precision_score(test_data.y_is_speech, predictions)])

    print()
    print(tabulate(table))

    # print the positive classifications
    if False:
        print()
        print('Speeches:')
        positives = test_data.X[predictions > 0.5, :]
        for i in range(positives.shape[0]):
            indices = positives[i, :]
            line = ' '.join(vocab.idx_to_token[idx] for idx in indices if idx in vocab.idx_to_token)
            print(line)


if __name__ == '__main__':
    main()
