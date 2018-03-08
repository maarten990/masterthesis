"""
Train the neural networks.

Usage:
train.py <paramfile> <folder> <files>... [options]
train.py (-h | --help)

Options:
    -h --help                      Show this screen
    --with_labels                  Include computed cluster labels.
    -b <size> --batch_size=<size>  The batch size used for training.
"""


import os.path
import re
import yaml
from collections import namedtuple
from docopt import docopt
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from tabulate import tabulate
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from data import GermanDataset, get_iterator, to_tensors
from models import LSTMClassifier, CNNClassifier, NameClassifier, WithClusterLabels

Datatuple = namedtuple('Datatuple', ['X_is_speech', 'X_speaker', 'y_is_speech', 'Y_speaker'])


class CNNParams:
    def __init__(self, embed_size: int, dropout: float, epochs: int,
                 num_filters: int) -> None:
        self.embed_size = embed_size
        self.dropout = dropout
        self.epochs = epochs
        self.num_filters = num_filters


class RNNParams:
    def __init__(self, embed_size: int, dropout: float, epochs: int,
                 num_layers: int, hidden_size: int) -> None:
        self.embed_size = embed_size
        self.dropout = dropout
        self.epochs = epochs
        self.num_layers = num_layers
        self.hidden_size = hidden_size


def get_filename(network: str, pattern: str) -> str:
    """
    Get the filename to use for pickling a classifier.

    network: the type of network (i.e. cnn, rnn or speaker)
    pattern: the glob pattern used for gathering training data

    Returns: a string representing a file path
    """
    path = f'pickle/{network}_{pattern}'
    path = re.sub(r'[*.]', '', path) + '.pkl'
    return path


def load_model(filename: str) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Load a model from the given path.
    """
    if os.path.exists(filename):
        modelfn, optimfn, model_state, optim_state = torch.load(filename)
        model = modelfn()
        optim = optimfn(model.parameters())

        model.load_state_dict(model_state)
        optim.load_state_dict(optim_state)
        return model, optim
    else:
        return None


def train(model: nn.Module, optimizer: torch.optim.Optimizer,
          dataloader: DataLoader, epochs: int = 100):
    model.train()

    epoch_losses = []
    t = trange(epochs, desc='Training')
    for _ in t:
        epoch_loss = torch.zeros(1).float()

        for batch in dataloader:
            data = to_tensors(batch)
            for size, d in data.items():
                X = d['data']
                c = d['cluster_data']
                y = d['label']

                y_pred = model(X, c)
                loss = model.loss(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.data.cpu()

        loss = epoch_loss[0]
        epoch_losses.append(loss)
        loss_delta = epoch_losses[-1] - epoch_losses[-2] if len(epoch_losses) > 1 else 0
        t.set_postfix({'loss': loss,
                       'Δloss': loss_delta})

    model.eval()
    return epoch_losses, optimizer


def evaluate_clf(model, Xb, cb, yb, batch_size=32, silent=False):
    """Evaluate the trained model.
    
    :param model: A trained model.
    :param Xb: A bucketed list of input data.
    :param cb: A bucketed list of cluster data.
    :param yb: A bucketed list of labels.
    :returns: A tuple of (precision, recall, f1 score).
    """
    model = model.eval()
    predictions = []
    true = []

    for X, c, y in zip(Xb, cb, yb):
        for i in range(0, X.shape[0], batch_size):
            Xvar = Variable(torch.from_numpy(X[i:i+32, :])).long()
            ybatch = y[i:i+32]
            cvar = Variable(torch.from_numpy(c[i:i+32, :])).float()

            pred = model(Xvar, cvar)
            pred = pred.cpu().squeeze().data.numpy()
            pred = np.where(pred > 0.5, 1, 0)
            predictions.extend(pred)
            true.extend(ybatch)

    table = []
    f1 = f1_score(true, predictions)
    p = precision_score(true, predictions)
    r = recall_score(true, predictions)
    table.append(['f1', f1])
    table.append(['Speech recall', r])
    table.append(['Speech precision', p])

    if not silent:
        print()
        print(tabulate(table))

    return p, r, f1


def setup_and_train(params: Union[CNNParams, RNNParams], with_labels: bool,
                    folder: str, files: List[str], batch_size: int = 32) -> List[float]:
    dataset = GermanDataset(folder, files, 5, 3, 1)

    recurrent_model: nn.Module
    if type(params) == RNNParams:
        cast(RNNParams, params)
        buckets = [5, 10, 15, 25, 40, -1]
        argdict = {'input_size': len(dataset.vocab.token_to_idx) + 1,
                   'embed_size': params.embed_size,
                   'hidden_size': params.hidden_size,
                   'num_layers': params.num_layers,
                   'dropout': params.dropout,
                   'use_final_layer': not with_labels}
        recurrent_model = LSTMClassifier(**argdict)
    elif type(params) == CNNParams:
        cast(CNNParams, params)
        buckets = [40]
        argdict = {'input_size': len(dataset.vocab.token_to_idx) + 1,
                   'seq_len': buckets[0],
                   'embed_size': params.embed_size,
                   'num_filters': params.num_filters,
                   'dropout': params.dropout,
                   'use_final_layer': not with_labels}
        recurrent_model = CNNClassifier(**argdict)

    model = WithClusterLabels(recurrent_model, 5, with_labels)
    optimizer = torch.optim.Adam(model.parameters())
    data = get_iterator(dataset, buckets=buckets, batch_size=batch_size)
    losses, optim = train(model, optimizer, data)

    #evaluate_clf(model, Xtb, ctb, ytb)
    #evaluate_spkr(model, Xtb, ytb, vocab.idx_to_token)

    return losses

def parse_params(params: Dict[str, Any]) -> Union[CNNParams, RNNParams, None]:
    """Parse a parameter dictinary and ensure it's valid.

    :param params: The parameter dict.
    :returns: The parsed parameters.
    """
    tp = params['type']
    del params['type']

    constructor: Union[Type[CNNParams],Type[RNNParams]]
    if tp == 'cnn':
        keys = ['embed_size', 'num_filters', 'dropout', 'epochs']
        constructor = CNNParams
    elif tp == 'rnn':
        keys = ['embed_size', 'hidden_size', 'num_layers', 'dropout', 'epochs']
        constructor = RNNParams
    else:
        print('Error: only cnn or rnn allowed as type.')
        return None

    for key in keys:
        if key not in params.keys():
            print(f'Error: missing key {key}')
            return None

    return constructor(**params)


if __name__ == '__main__':
    args = docopt(__doc__)
    paramfile = args['<paramfile>']
    folder = args['<folder>']
    files = args['<files>']
    with_labels = args['--with_labels']
    batch_size = args['--batch_size']

    with open(paramfile, 'r') as f:
        params = yaml.load(f)
        p = parse_params(params)

        losses = setup_and_train(p, with_labels, folder, files, int(batch_size))
        print(losses)
