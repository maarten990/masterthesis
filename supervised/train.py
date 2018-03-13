"""
Train the neural networks.

Usage:
train.py <paramfile> <files>... [options]
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

from data import GermanDatasetInMemory, get_iterator, to_tensors
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
          dataloader: DataLoader, epochs: int = 100) -> List[float]:
    """Train a Pytorch model.

    :param model: A Pytorch model.
    :param optimizer: A Pytorch optimizer for the model.
    :param dataloader: An iterator returning batches.
    :param epochs: The number of epochs to train for.
    :returns: The value of the model's loss function at every epoch.
    """
    model.train()

    epoch_losses: List[float] = []
    best_params: Dict[str, Any] = {}
    best_loss = 99999

    t = trange(epochs, desc='Training')
    for _ in t:
        epoch_loss = torch.zeros(1).float()

        for batch in dataloader:
            data = to_tensors(batch)
            for _, d in data.items():
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

        # check if the model is the best yet
        if loss < best_loss:
            best_loss = loss
            best_params = model.state_dict()

        # update the progress bar
        loss_delta = epoch_losses[-1] - epoch_losses[-2] if len(epoch_losses) > 1 else 0
        t.set_postfix({'loss': loss,
                       'Δloss': loss_delta})

    t.close()

    model.load_state_dict(best_params)
    model.eval()
    return epoch_losses


def evaluate_clf(model: nn.Module, dataloader: DataLoader, cutoff: float = 0.5,
                 silent=False) -> Tuple[float, float, float]:
    """Evaluate the trained model.
    
    :param model: A trained model.
    :param dataloader: The input data.
    :param cutoff: The value (between 0 and 1) from which point the neural
        network output is considered positive.
    :param silent: If True, don't print the scores.
    :returns: A tuple of (precision, recall, f1 score).
    """
    model = model.eval()
    predictions: List[bool] = []
    true: List[bool] = []

    for batch in dataloader:
        data = to_tensors(batch)
        for _, d in data.items():
            X = d['data']
            c = d['cluster_data']
            y = d['label']

            pred = model(X, c)
            pred = pred.cpu().squeeze().data.numpy()
            pred = np.where(pred > cutoff, 1, 0)
            predictions.extend(pred)
            true.extend(y.data.numpy())

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
                    dataset: Dataset, batch_size: int = 32) -> Tuple[nn.Module, List[float]]:
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

    data = get_iterator(dataset, buckets=buckets, batch_size=batch_size)
    model = WithClusterLabels(recurrent_model, data.dataset.num_clusterlabels,
                              with_labels)
    optimizer = torch.optim.Adam(model.parameters())
    losses = train(model, optimizer, data)

    return model, losses

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
    files = args['<files>']
    with_labels = args['--with_labels']
    batch_size = args['--batch_size']

    with open(paramfile, 'r') as f:
        params = yaml.load(f)
        p = parse_params(params)

        losses = setup_and_train(p, with_labels, files, int(batch_size))
        print(losses)
