
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
        return hidden.cuda()

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)
