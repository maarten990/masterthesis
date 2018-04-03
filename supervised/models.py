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
    def __init__(self, input_size, seq_len, embed_size, num_filters, dropout,
                 use_final_layer=True):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embed_size)
        self.pool = nn.MaxPool1d(2)
        self.conv = nn.Conv1d(embed_size, num_filters, 3)

        clf_size = self.pool(
            self.conv(Variable(torch.zeros(32, embed_size, seq_len)))).size(2) * num_filters
        self.clf_h = nn.Linear(1900, int(clf_size / 2))
        self.clf_out = nn.Linear(int(clf_size / 2), 1)

        self.use_final_layer = use_final_layer
        self.output_size = 1 if self.use_final_layer else clf_size

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, inputs):
        embedded = self.embedding(inputs)

        # permute from [batch, seq_len, input_size] to [batch, input_size, seq_len]
        embedded = embedded.permute(0, 2, 1)
        pooled = self.pool(self.conv(embedded))

        batch_size = inputs.size(0)
        clf_in = pooled.view(batch_size, -1)

        if self.use_final_layer:
            h = F.relu(self.dropout(self.clf_h(clf_in)))
            return F.sigmoid(self.dropout(self.clf_out(h)))
        else:
            return F.relu(clf_in)

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)


class LSTMClassifier(nn.Module):
    """ LSTM-based speech classifier. """
    def __init__(self, input_size, embed_size, hidden_size, num_layers, dropout,
                 use_final_layer=True):
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

        self.use_final_layer = use_final_layer
        self.output_size = 1 if self.use_final_layer else hidden_size * 2

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, inputs):
        # initialize the lstm hidden states
        hidden = self.init_hidden(inputs.size(0))
        cell = self.init_hidden(inputs.size(0))

        # run the LSTM over the full input sequence and take the average over
        # all the outputs
        embedded = self.embedding(inputs)
        outputs, _ = self.rnn(embedded, (hidden, cell))
        averaged = torch.mean(outputs, dim=1)

        if self.use_final_layer:
            # sigmoid classification with 1 hidden layer in between
            hiddenlayer = F.relu(self.dropout(self.clf_h(averaged)))
            return F.sigmoid(self.dropout(self.clf_out(hiddenlayer)))
        else:
            return averaged

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
    def __init__(self, recurrent_clf, n_labels, use_labels, dropout):
        super().__init__()
        self.recurrent_clf = recurrent_clf
        self.use_labels = use_labels

        if use_labels:
            output_size = self.recurrent_clf.output_size
            self.dropout = nn.Dropout(dropout)
            self.linear1 = nn.Linear(output_size + n_labels, int(output_size / 2))
            self.linear2 = nn.Linear(int(output_size / 2), 1)

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, inputs, labels):
        recurrent_output = self.recurrent_clf(inputs)

        if not self.use_labels:
            return recurrent_output
        else:
            combined = torch.cat([recurrent_output, labels], 1)
            h = F.relu(self.dropout(self.linear1(combined)))
            out = self.dropout(self.linear2(h))
            return F.sigmoid(out)

    def loss(self, y_pred, y_true):
        return self.recurrent_clf.loss(y_pred, y_true)
