import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNNClassifier(nn.Module):
    """ CNN-based speech classifier. """

    def __init__(
        self,
        input_size,
        seq_len,
        embed_size,
        filters,
        dropout,
        num_layers=1,
        batch_norm=False,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embed_size)
        self.layers = nn.ModuleList([])
        self.layers.append(
            nn.ModuleList([nn.Conv1d(embed_size, num, size, padding=(size - 1) / 2)
                           for num, size in filters])
        )

        num_filters = sum([num for num, _ in filters])
        for i in range(num_layers - 1):
            self.layers.append(
                nn.ModuleList(
                    [nn.Conv1d(num_filters, num, size, padding=(size - 1) / 2) for num, size in filters]
                )
            )

        self.batch_norm = batch_norm
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(num) for num, _ in filters])

        # calculate classifier size
        temp_data = Variable(torch.zeros((1, embed_size, seq_len)))
        for layer in self.layers:
            d = [conv(temp_data) for conv in layer]
            temp_data = torch.cat(d, 1)

        temp_data = F.max_pool1d(temp_data, kernel_size=temp_data.shape[2])
        self.output_size = temp_data.view(1, -1).shape[1]

    def forward(self, inputs):
        embedded = self.embedding(inputs)

        # permute from [batch, seq_len, input_size] to [batch, input_size, seq_len]
        data = embedded.permute(0, 2, 1)

        for layer in self.layers:
            d = [F.relu(conv(data)) for conv in layer]
            if self.batch_norm:
                d = [bn(l) for bn, l in zip(self.batch_norms, d)]
            data = torch.cat(d, 1)

        data = F.max_pool1d(data, kernel_size=data.shape[2])
        batch_size = inputs.size(0)
        return F.relu(data.view(batch_size, -1))

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)


class LSTMClassifier(nn.Module):
    """ LSTM-based speech classifier. """

    def __init__(self, input_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # the output size of the rnn is 2 * hidden_size because it's bidirectional
        self.output_size = hidden_size * 2

    def forward(self, inputs):
        # initialize the lstm hidden states
        hidden = self.init_hidden(inputs.size(0))
        cell = self.init_hidden(inputs.size(0))

        # run the LSTM over the full input sequence and take the average over
        # all the outputs
        embedded = self.embedding(inputs)
        output, _ = self.rnn(embedded, (hidden, cell))
        return output[:, -1, :]

    def init_hidden(self, batch_size):
        "Initialize a zero hidden state with the appropriate dimensions."
        hidden = Variable(torch.zeros(1, self.hidden_size))
        hidden = hidden.repeat(self.num_layers * 2, batch_size, 1)

        if torch.cuda.is_available():
            hidden = hidden.cuda()

        return hidden

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)


class NoClusterLabels(nn.Module):

    def __init__(self, recurrent_clf, dropout, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.recurrent_clf = recurrent_clf
        self.recurrent_clf.batch_norm = batch_norm

        output_size = self.recurrent_clf.output_size
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(output_size, int(output_size / 2))
        self.bn = nn.BatchNorm1d(int(output_size / 2))
        self.linear2 = nn.Linear(int(output_size / 2), 1)

    def forward(self, inputs, labels):
        recurrent_output = self.recurrent_clf(inputs)

        h = self.linear1(recurrent_output)
        if self.batch_norm:
            h = self.bn(h)
        h = F.relu(self.dropout(h))
        out = self.linear2(h)
        return F.sigmoid(out)

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)


class CategoricalClusterLabels(nn.Module):

    def __init__(self, recurrent_clf, n_labels, window_size, dropout, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.recurrent_clf = recurrent_clf
        self.recurrent_clf.batch_norm = batch_norm

        output_size = self.recurrent_clf.output_size + (n_labels * window_size)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(output_size, int(output_size / 2))
        self.bn = nn.BatchNorm1d(int(output_size / 2))
        self.linear2 = nn.Linear(int(output_size / 2), 1)

    def forward(self, inputs, labels):
        recurrent_output = self.recurrent_clf(inputs)
        combined = torch.cat([recurrent_output, labels.float()], 1)
        h = self.linear1(combined)
        if self.batch_norm:
            h = self.bn(h)
        h = F.relu(self.dropout(h))
        out = self.linear2(h)
        return F.sigmoid(out)

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)


class CategoricalClusterLabelsOnlyCenter(nn.Module):

    def __init__(self, recurrent_clf, n_labels, dropout, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.recurrent_clf = recurrent_clf
        self.recurrent_clf.batch_norm = batch_norm

        output_size = self.recurrent_clf.output_size + n_labels
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(output_size, int(output_size / 2))
        self.bn = nn.BatchNorm1d(int(output_size / 2))
        self.linear2 = nn.Linear(int(output_size / 2), 1)

    def forward(self, inputs, labels):
        recurrent_output = self.recurrent_clf(inputs)
        combined = torch.cat([recurrent_output, labels.float()], 1)
        h = self.linear1(combined)
        if self.batch_norm:
            h = self.bn(h)
        h = F.relu(self.dropout(h))
        out = self.linear2(h)
        return F.sigmoid(out)

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)


class ClusterLabelsCNN(nn.Module):

    def __init__(self, recurrent_clf, n_labels, dropout, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.recurrent_clf = recurrent_clf
        self.recurrent_clf.batch_norm = batch_norm
        self.label_cnn = CNNClassifier(
            n_labels,
            5,
            n_labels,
            [(32, 1), (32, 2), (32, n_labels)],
            dropout,
            num_layers=1,
        )

        output_size = self.recurrent_clf.output_size + self.label_cnn.output_size
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(output_size, int(output_size / 2))
        self.bn = nn.BatchNorm1d(int(output_size / 2))
        self.linear2 = nn.Linear(int(output_size / 2), 1)

    def forward(self, inputs, labels):
        recurrent_output = self.recurrent_clf(inputs)
        label_output = self.label_cnn(labels)
        combined = torch.cat([recurrent_output, label_output], 1)
        h = self.linear1(combined)
        if self.batch_norm:
            h = self.bn(h)
        h = F.relu(self.dropout(h))
        out = self.linear2(h)
        return F.sigmoid(out)

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)


class ClusterLabelsRNN(nn.Module):

    def __init__(self, recurrent_clf, n_labels, dropout, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.recurrent_clf = recurrent_clf
        self.recurrent_clf.batch_norm = batch_norm
        self.label_cnn = LSTMClassifier(n_labels, n_labels, 100, 1, dropout)

        output_size = self.recurrent_clf.output_size + self.label_cnn.output_size
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(output_size, int(output_size / 2))
        self.bn = nn.BatchNorm1d(int(output_size / 2))
        self.linear2 = nn.Linear(int(output_size / 2), 1)

    def forward(self, inputs, labels):
        recurrent_output = self.recurrent_clf(inputs)
        label_output = self.label_cnn(labels)
        combined = torch.cat([recurrent_output, label_output], 1)
        h = self.linear1(combined)
        if self.batch_norm:
            h = self.bn(h)
        h = F.relu(self.dropout(h))
        out = self.linear2(h)
        return F.sigmoid(out)

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)
