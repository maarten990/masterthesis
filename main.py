import argparse
from collections import namedtuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from data import sliding_window
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from keras.preprocessing.sequence import pad_sequences

from tabulate import tabulate

Split = namedtuple('Split', ['X_train', 'X_test', 'y_train', 'y_test'])


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.clf_h = nn.Linear(hidden_size * 2, hidden_size)
        self.clf_out = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        "Perform a full pass of the encoder over the entire input."
        hidden = self.init_hidden(inputs.size()[0])
        cell = self.init_hidden(inputs.size()[0])

        embedded = self.dropout(self.embedding(inputs))
        outputs, _ = self.rnn(embedded, (hidden, cell))

        averaged = torch.mean(outputs, dim=1).squeeze()
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


def get_data(args):
    X, y, vocab = sliding_window(args.folder, args.pattern, 2, 0.01)
    X = pad_sequences(X)

    split = Split(*train_test_split(X, y, test_size=args.test_size))

    print('{} training samples, {} testing samples'.format(
        split.X_train.shape[0], split.X_test.shape[0]))
    print('Number of features: {}'.format(split.X_train.shape[1]))

    return split, vocab


def train(model, X_train, y_train, epochs=100, batch_size=32):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
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

        print(f'Epoch {epoch}: loss {epoch_loss.data[0]:.3f}')



def main():
    args = get_args()

    split, vocab = get_data(args)
    print(split.X_train.shape)
    model = LSTMClassifier(len(vocab.token_to_idx), 128, 32, 1)

    train(model, split.X_train, split.y_train, args.epochs)
    model.train(False)

    predictions = model(Variable(torch.from_numpy(split.X_test)).long())
    predictions = predictions.data.numpy()
    predictions = np.where(predictions > 0.5, 1, 0)
    print()

    table = []
    table.append(['Accuracy', accuracy_score(split.y_test, predictions)])
    table.append(['f1', f1_score(split.y_test, predictions)])
    table.append(['Speech recall', recall_score(split.y_test, predictions)])
    table.append(['Speech precision', precision_score(split.y_test, predictions)])

    print()
    print(tabulate(table))


if __name__ == '__main__':
    main()
