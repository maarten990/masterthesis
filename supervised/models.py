import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def cnn_output_size(cnn: nn.Conv1d) -> int:
    numerator = (
        cnn.in_channels
        + 2 * cnn.padding[0]
        - cnn.dilation[0] * (cnn.kernel_size[0] - 1)
        - 1
    )
    return (numerator // cnn.stride[0]) + 1


class Conv1dMultipleFilters(nn.Module):

    def __init__(self, in_channels, kernel_sizes):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels, num, size, padding=(size - 1) / 2)
                for num, size in kernel_sizes
            ]
        )

        self.output_size = int(sum([cnn_output_size(conv) for conv in self.convs]))

    def forward(self, inputs):
        filters = [conv(inputs) for conv in self.convs]
        return torch.cat(filters, 1)


class Pool1Max(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return F.max_pool1d(inputs, kernel_size=inputs.shape[2])


class TransposeEmbed(nn.Module):
    """
    Permute the embedding output from from [batch, seq_len, input_size] to
    [batch, input_size, seq_len].
    """

    def __init__(self, input_size, embed_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embed_size)

    def forward(self, inputs):
        return self.embedding(inputs).permute(0, 2, 1)


class CNNClassifier(nn.Module):
    """ CNN-based speech classifier. """

    def __init__(self, input_size, seq_len, embed_size, filters):
        super().__init__()

        self.network = nn.Sequential(
            TransposeEmbed(input_size, embed_size),
            Conv1dMultipleFilters(embed_size, filters),
            Pool1Max(),
            nn.ReLU(),
        )
        self.output_size = sum([num for num, _ in filters])

    def forward(self, inputs):
        out = self.network(inputs)
        batch_size = inputs.size(0)
        return F.relu(out.view(batch_size, -1))

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


class ClfBase(nn.Module):

    def __init__(self, dropout, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(output_size, output_size // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(output_size // 2, 1),
            nn.Sigmoid(),
        )

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)


class NoClusterLabels(ClfBase):

    def __init__(self, recurrent_clf, dropout):
        super().__init__(dropout, recurrent_clf.output_size)
        self.recurrent_clf = recurrent_clf

    def forward(self, inputs, labels):
        return self.network(self.recurrent_clf(inputs))


class CategoricalClusterLabels(ClfBase):

    def __init__(self, recurrent_clf, n_labels, window_size, dropout, batch_norm=False):
        super().__init__(dropout, recurrent_clf.output_size + (n_labels * window_size))
        self.recurrent_clf = recurrent_clf

    def forward(self, inputs, labels):
        r_out = self.recurrent_clf(inputs)
        combined = torch.cat([r_out, labels.float()], 1)
        return self.network(combined)


class CategoricalClusterLabelsOnlyCenter(ClfBase):

    def __init__(self, recurrent_clf, n_labels, dropout, batch_norm=False):
        super().__init__(dropout, recurrent_clf.output_size + n_labels)
        self.recurrent_clf = recurrent_clf

    def forward(self, inputs, labels):
        r_out = self.recurrent_clf(inputs)
        combined = torch.cat([r_out, labels.float()], 1)
        return self.network(combined)


class ClusterLabelsCNN(ClfBase):

    def __init__(self, recurrent_clf, n_labels, dropout, batch_norm=False):
        label_cnn = CNNClassifier(
            n_labels, 5, n_labels, [(100, 3)], dropout, num_layers=1
        )
        super().__init__(
            dropout, recurrent_clf.output_size + self.label_cnn.output_size
        )
        self.label_cnn = label_cnn
        self.recurrent_clf = recurrent_clf

    def forward(self, inputs, labels):
        r_out = self.recurrent_clf(inputs)
        l_out = self.label_cnn(labels)
        combined = torch.cat([r_out, l_out], 1)
        return self.network(combined)


class ClusterLabelsRNN(ClfBase):

    def __init__(self, recurrent_clf, n_labels, dropout, batch_norm=False):
        label_rnn = LSTMClassifier(n_labels, n_labels, 100, 1, dropout)
        super().__init__(dropout, recurrent_clf.output_size + label_rnn.output_size)
        self.label_rnn = label_rnn
        self.recurrent_clf = recurrent_clf

    def forward(self, inputs, labels):
        r_out = self.recurrent_clf(inputs)
        l_out = self.label_rnn(labels)
        combined = torch.cat([r_out, l_out], 1)
        return self.network(combined)
