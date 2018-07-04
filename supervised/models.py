import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CharCNN(nn.Module):

    def __init__(self, input_size, seq_len, *args):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_size, 256, 7, padding=0),
            nn.MaxPool1d(3),
            nn.ReLU(),
            nn.Conv1d(256, 256, 7, padding=0),
            nn.MaxPool1d(3),
            nn.ReLU(),

            nn.Conv1d(256, 256, 3, padding=0),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=0),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=0),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=0),
            Pool1Max(),
            nn.ReLU(),
        )

        self.input_size = input_size
        self.output_size = 256

    def forward(self, inputs):
        onehot = torch.Tensor(np.eye(self.input_size,
                                     dtype='uint8')[inputs]).float().cuda()
        onehot = onehot.permute(0, 2, 1)
        out = self.network(onehot)
        batch_size = inputs.size(0)
        return out.view(batch_size, -1)

    def loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true)


class ClfBase(nn.Module):

    def __init__(self, dropout, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(output_size, 1024),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(1024, 1),
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


class ClusterLabelsChar(ClfBase):

    def __init__(self, recurrent_clf, n_labels, window_size, dropout, batch_norm=False):
        super().__init__(dropout, recurrent_clf.output_size + (n_labels * window_size))
        self.recurrent_clf = recurrent_clf

    def forward(self, inputs, labels):
        r_out = self.recurrent_clf(inputs)
        combined = torch.cat([r_out, labels.float()], 1)
        return self.network(combined)
