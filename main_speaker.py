import argparse
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

from data import speaker_timeseries

Split = namedtuple('Split', ['X_train', 'X_test', 'y_train', 'y_test'])


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers,
                           bidirectional=True, batch_first=True)

    def forward(self, inputs):
        "Perform a full pass of the encoder over the entire input."
        hidden = self.init_hidden(inputs.size()[0])
        cell = self.init_hidden(inputs.size()[0])

        embedded = self.embedding(inputs)
        _, (hidden, _) = self.rnn(embedded, (hidden, cell))

        return hidden

    def init_hidden(self, batch_size):
        "Initialize a zero hidden state with the appropriate dimensions."
        hidden = Variable(torch.zeros(1, self.hidden_size))
        hidden = hidden.repeat(self.num_layers * 2, batch_size, 1)
        return hidden


class NameClassifier(nn.Module):
    def __init__(self, input_size, seq_length, embed_size, encoder_hidden, num_layers=1):
        super().__init__()

        self.encoder = Encoder(input_size, embed_size, encoder_hidden, num_layers)

        n_classif_hidden = int(((2 * encoder_hidden) + seq_length) / 2)
        self.out_hidden = nn.Linear(2 * encoder_hidden, n_classif_hidden)
        self.out_classif = nn.Linear(n_classif_hidden, seq_length)

    def forward(self, input, force_teacher=False):
        # first encode the input sequence and get the output of the final layer
        hidden_enc = self.encoder(input)
        context_vector = torch.cat((hidden_enc[-2, :, :], hidden_enc[-1, :, :]), 1)

        # use it to classify whether each input word is part of the name
        h = self.out_hidden(context_vector)
        h = F.relu(h)
        out = self.out_classif(h)
        out = F.sigmoid(out)

        return out


def get_data(args):
    X, y, char_to_idx, idx_to_char = speaker_timeseries(args.parsed_folder, args.pattern)
    print(X.shape)

    split = Split(*train_test_split(X, y, test_size=args.test_size))

    print('{} training samples, {} testing samples'.format(
        split.X_train.shape[0], split.X_test.shape[0]))
    print('Number of features: {}'.format(split.X_train.shape[1]))

    return split, char_to_idx, idx_to_char


def train(model, X_train, y_train, epochs=100, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        epoch_loss = Variable(torch.zeros(1)).float()

        for i in range(0, X_train.shape[0], batch_size):
            X = Variable(torch.from_numpy(X_train[i:i+32, :])).long()
            y = Variable(torch.from_numpy(y_train[i:i+32, :])).float()

            y_pred = model(X)

            # calculate the loss as the binary cross entropy between the
            # flattened versions of the arrays
            loss = F.binary_cross_entropy(y_pred.view(-1), y.view(-1))
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}: loss {epoch_loss.data[0]:.3f}')


def test(model, X_test, y_test, idx_to_char):
    predictions = model(Variable(torch.from_numpy(X_test)).long())

    rows = []
    for i in range(X_test.shape[0]):
        full_string = X_test[i, :]
        true = y_test[i, :]
        pred = predictions.data.numpy()[i, :]

        true_words = full_string[true > 0.5]
        pred_words = full_string[pred > 0.5]

        full_input = ' '.join([idx_to_char[idx] for idx in full_string if idx != 0])
        true_sent = ' '.join([idx_to_char[idx] for idx in true_words if idx != 0])
        pred_sent = ' '.join([idx_to_char[idx] for idx in pred_words if idx != 0])

        row = [full_input, true_sent, pred_sent]
        rows.append(row)

    print(tabulate(rows, headers=['Input', 'true', 'predicted']))


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

    return parser.parse_args()


def main():
    args = get_args()

    split, char_to_idx, idx_to_char = get_data(args)
    model = NameClassifier(input_size=len(char_to_idx) + 1,  # offset by 1 because 0 is not included
                           seq_length=split.X_train.shape[1],
                           embed_size=128,
                           encoder_hidden=64,
                           num_layers=1)

    train(model, split.X_train, split.y_train, epochs=args.epochs)
    test(model, split.X_test, split.y_test, idx_to_char)


if __name__ == '__main__':
    main()
