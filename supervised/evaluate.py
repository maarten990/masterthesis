from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from tabulate import tabulate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data import get_iterator, to_cpu, to_gpu, to_tensors

sns.set()


def evaluate_clf(model: nn.Module, dataloader: DataLoader, cutoff: float = 0.5,
                 silent: bool=False, gpu: bool=True) -> Tuple[float, float]:
    """Evaluate the trained model.

    :param model: A trained model.
    :param dataloader: The input data.
    :param cutoff: The value (between 0 and 1) from which point the neural
        network output is considered positive.
    :param silent: If True, don't print the scores.
    :param gpu: If true, run on the gpu. Otherwise use the cpu.
    :returns: A tuple of (precision, recall).
    """
    model = model.eval()
    model.cuda() if gpu else model.cpu()

    predictions: List[bool] = []
    true: List[bool] = []

    for batch in dataloader:
        data = to_tensors(batch)
        if gpu:
            data = to_gpu(data)

        for _, d in data.items():
            X = d['data']
            c = d['cluster_data']
            y = d['label']

            pred = model(X, c)
            pred = pred.cpu().squeeze().data.numpy()
            pred = np.where(pred > cutoff, 1, 0)
            predictions.extend(pred)
            true.extend(y.data.cpu().numpy())

    table = []
    if not 1 in predictions:
        p = 1.0
    else:
        p = precision_score(true, predictions)
    r = recall_score(true, predictions)
    table.append(['Speech recall', r])
    table.append(['Speech precision', p])

    if not silent:
        print()
        print(tabulate(table))

    return p, r


def precision_recall_values(model: nn.Module, dataloader: DataLoader, gpu: bool=True) -> Tuple[List[float], List[float]]:
    """Calculate the values for a  precision-recall curve by varying the classification cutoff.

    :param model: A trained model.
    :param dataloader: The input data.
    :param gpu: If true, run on the gpu. Otherwise use the cpu.
    :returns: A list of (precision, recall) tuples, sorted by increasing recall.
    """
    pr: List[Tuple[float, float]] = []
    for cutoff in np.linspace(0, 1):
        p, r = evaluate_clf(model, dataloader, cutoff, silent=True, gpu=gpu)
        pr.append((p, r))

    # sort the values by recall
    pr = sorted(pr, key=lambda x: x[1])

    # split the tuples into the 2 lists to return
    p, r = np.array(pr).T
    return p, r


def average_precision(precision: List[float], recall: List[float]) -> float:
    return np.trapz(precision, recall)


def plot(curves: Dict[str, Union[List[float], Tuple[List[float], List[float]]]], xlabel: str, ylabel: str,
         monotonic: bool = False, title: str = ''):
    """Plot a number of curves.

    :param curves: A dictionary mapping the label of the plot to either its x
        and y values, or just the y values.
    :param xlabel: The label for the x-axis.
    :param ylabel: The label for the y-axis.
    :param monotonic: Make the plot monotonically increasing.
    :param title: Optional title to add to the plot."""
    for label, values in curves.items():
        if isinstance(values, tuple):
            x, y = values
        else:
            x = list(range(len(values)))
            y = values
        if monotonic:
            for i in range(1, len(y)):
                y[i] = min(y[i], y[i-1])

        plt.plot(x, y, label=label)

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title:
        plt.title(title)
