import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
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

        if torch.cuda.is_available():
            self.cuda()

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

        if torch.cuda.is_available():
            hidden = hidden.cuda()

        return hidden


class NameClassifier(nn.Module):
    """
    LSTM-based classifier that outputs a boolean array indicating wether each
    input word belongs to the set of words denoting the speaker.
    """
    def __init__(self, input_size, seq_length, embed_size, encoder_hidden, dropout, num_layers=1):
        super().__init__()

        self.encoder = Encoder(input_size, embed_size, encoder_hidden, num_layers, dropout)
        self.dropout = nn.Dropout(dropout)

        n_classif_hidden = int(((2 * encoder_hidden) + seq_length) / 2)
        self.out_hidden = nn.Linear(2 * encoder_hidden, n_classif_hidden)
        self.out_classif = nn.Linear(n_classif_hidden, seq_length)

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, input, force_teacher=False):
        # first encode the input sequence and get the output of the final layer
        hidden_enc = self.encoder(input)
        context_vector = torch.cat((hidden_enc[-2, :, :], hidden_enc[-1, :, :]), 1)

        # use it to classify whether each input word is part of the name
        h = self.dropout(self.out_hidden(context_vector))
        h = self.dropout(F.relu(h))
        out = self.out_classif(h)
        out = F.sigmoid(out)

        return out

    def loss(self, y_pred, y_true):
        # calculate the loss as the binary cross entropy between the
        # flattened versions of the arrays
        return F.binary_cross_entropy(y_pred.view(-1), y_true.view(-1))


class CNNClassifier(nn.Module):
    """ CNN-based speech classifier. """
    def __init__(self, input_size, seq_len, embed_size, filters, dropout,
                 num_layers=1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embed_size)
        self.layers = nn.ModuleList([])

        self.layers.append(nn.ModuleList([nn.Conv1d(embed_size, num, size) for num, size in filters]))
        for i in range(num_layers - 1):
            self.layers.append(nn.ModuleList([nn.Conv1d(embed_size, num, size) for num, size in filters]))

        # calculate classifier size
        temp_data = Variable(torch.zeros((1, embed_size, seq_len)))
        for layer in self.layers:
            d = [conv(temp_data) for conv in layer]
            d = [F.max_pool1d(l, kernel_size=l.shape[2]) for l in d]
            temp_data = torch.cat(d, 1)

        temp_data = F.max_pool1d(temp_data, kernel_size=temp_data.shape[2])
        self.output_size = temp_data.view(1, -1).shape[1]

    def forward(self, inputs):
        embedded = self.embedding(inputs)

        # permute from [batch, seq_len, input_size] to [batch, input_size, seq_len]
        data = embedded.permute(0, 2, 1)

        for layer in self.layers:
            d = [conv(data) for conv in layer]
            d = [F.max_pool1d(l, kernel_size=l.shape[2]) for l in d]
            data = torch.cat(d, 1)

        # data = F.max_pool1d(data, kernel_size=data.shape[2])
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
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers,
                           bidirectional=True, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)

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


class WithClusterLabels(nn.Module):
    def __init__(self, recurrent_clf, n_labels, use_labels, dropout, only_labels=False):
        super().__init__()
        self.recurrent_clf = recurrent_clf
        self.use_labels = use_labels
        self.only_labels = only_labels
        self.label_cnn = CNNClassifier(n_labels, 5, 2 * n_labels, [(16, 3), (16, 4), (16, 5)], dropout,
                                       num_layers=1, use_final_layer=False)

        if only_labels:
            self.dropout = nn.Dropout(dropout)
            self.linear1 = nn.Linear(n_labels, int(n_labels / 2))
            self.linear2 = nn.Linear(int(n_labels / 2), 1)
        elif use_labels:
            output_size = self.recurrent_clf.output_size + self.label_cnn.output_size
            self.dropout = nn.Dropout(dropout)
            self.linear1 = nn.Linear(output_size, int(output_size / 2))
            self.linear2 = nn.Linear(int(output_size / 2), 1)

    def forward(self, inputs, labels):
        if self.only_labels:
            h = F.relu(self.dropout(self.linear1(labels)))
            out = self.dropout(self.linear2(h))
            return F.sigmoid(out)

        recurrent_output = self.recurrent_clf(inputs)

        if not self.use_labels:
            return recurrent_output
        else:
            label_output = self.label_cnn(labels)
            combined = torch.cat([recurrent_output, label_output], 1)
            h = F.relu(self.dropout(self.linear1(combined)))
            out = self.dropout(self.linear2(h))
            return F.sigmoid(out)

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)


class NoClusterLabels(nn.Module):
    def __init__(self, recurrent_clf, dropout):
        super().__init__()
        self.recurrent_clf = recurrent_clf

        output_size = self.recurrent_clf.output_size
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(output_size, int(output_size / 2))
        self.linear2 = nn.Linear(int(output_size / 2), 1)

    def forward(self, inputs, labels):
        recurrent_output = self.recurrent_clf(inputs)

        h = F.relu(self.dropout(self.linear1(recurrent_output)))
        out = self.dropout(self.linear2(h))
        return F.sigmoid(out)

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)


class CategoricalClusterLabels(nn.Module):
    def __init__(self, recurrent_clf, n_labels, window_size, dropout):
        super().__init__()
        self.recurrent_clf = recurrent_clf

        output_size = self.recurrent_clf.output_size + (n_labels * window_size)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(output_size, int(output_size / 2))
        self.linear2 = nn.Linear(int(output_size / 2), 1)

    def forward(self, inputs, labels):
        recurrent_output = self.recurrent_clf(inputs)
        combined = torch.cat([recurrent_output, labels.float()], 1)
        h = F.relu(self.dropout(self.linear1(combined)))
        out = self.dropout(self.linear2(h))
        return F.sigmoid(out)

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)


class CategoricalClusterLabelsOnlyCenter(nn.Module):
    def __init__(self, recurrent_clf, n_labels, dropout):
        super().__init__()
        self.recurrent_clf = recurrent_clf

        output_size = self.recurrent_clf.output_size + n_labels
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(output_size, int(output_size / 2))
        self.linear2 = nn.Linear(int(output_size / 2), 1)

    def forward(self, inputs, labels):
        recurrent_output = self.recurrent_clf(inputs)
        combined = torch.cat([recurrent_output, labels.float()], 1)
        h = F.relu(self.dropout(self.linear1(combined)))
        out = self.dropout(self.linear2(h))
        return F.sigmoid(out)

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)


class ClusterLabelsCNN(nn.Module):
    def __init__(self, recurrent_clf, n_labels, dropout):
        super().__init__()
        self.recurrent_clf = recurrent_clf
        self.label_cnn = CNNClassifier(n_labels, 5, 2 * n_labels, [(16, 3), (16, 4), (16, 5)], dropout,
                                       num_layers=1)

        output_size = self.recurrent_clf.output_size + self.label_cnn.output_size
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(output_size, int(output_size / 2))
        self.linear2 = nn.Linear(int(output_size / 2), 1)

    def forward(self, inputs, labels):
        recurrent_output = self.recurrent_clf(inputs)
        label_output = self.label_cnn(labels)
        combined = torch.cat([recurrent_output, label_output], 1)
        h = F.relu(self.dropout(self.linear1(combined)))
        out = self.dropout(self.linear2(h))
        return F.sigmoid(out)

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)
