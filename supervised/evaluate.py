from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC
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


def evaluate_bow(model: SVC, vectorizer: TfidfVectorizer, dataset: Dataset,
                 cutoff: float = 0.5) -> Tuple[float, float]:
    samples = [list(entry.values())[0]['data'] for entry in dataset]
    true = [list(entry.values())[0]['label'] for entry in dataset]

    y = model.predict_proba(vectorizer.transform(samples))
    predictions = np.where(y > cutoff, 1, 0)
    predictions = np.argmax(predictions, axis=1)
    if not 1 in predictions:
        p = 1.0
    else:
        p = precision_score(true, predictions)
    r = recall_score(true, predictions)

    return p, r


def precision_recall_values(model: nn.Module, dataloader: DataLoader, gpu: bool=True) -> Tuple[List[float], List[float]]:
    """Calculate the values for a  precision-recall curve by varying the classification cutoff.

    :param model: A trained model.
    :param dataloader: The input data.
    :param gpu: If true, run on the gpu. Otherwise use the cpu.
    :returns: A list of (precision, recall) tuples, sorted by increasing recall.
    """
    pr: List[Tuple[float, float]] = []
    for cutoff in np.linspace(0, 1, num=25):
        p, r = evaluate_clf(model, dataloader, cutoff, silent=True, gpu=gpu)
        pr.append((p, r))

    # sort the values by recall
    pr = sorted(pr, key=lambda x: x[1])

    # split the tuples into the 2 lists to return
    p, r = np.array(pr).T
    return p, r


def average_precision(precision: List[float], recall: List[float]) -> float:
    return np.trapz(precision, recall)


def max_f1(precision: List[float], recall: List[float]) -> float:
    p = np.array(precision)
    r = np.array(recall)
    return np.max(2 * ((p * r) / (p + r)))


def plot(curves: Dict[str, Union[List[float], Tuple[List[float], List[float]]]], xlabel: str, ylabel: str,
         monotonic: bool = False, title: str = '') -> plt.Figure:
    """Plot a number of curves.

    :param curves: A dictionary mapping the label of the plot to either its x
        and y values, or just the y values.
    :param xlabel: The label for the x-axis.
    :param ylabel: The label for the y-axis.
    :param monotonic: Make the plot monotonically increasing.
    :param title: Optional title to add to the plot."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for label, values in curves.items():
        if isinstance(values, tuple):
            x, y = values
        else:
            x = list(range(len(values)))
            y = values
        if monotonic:
            for i in range(1, len(y)):
                y[i] = min(y[i], y[i-1])

        ax.plot(x, y, label=label)

    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    return fig


def compare(plain: nn.Module, with_clusters: nn.Module, dataset: Dataset
           ) -> Tuple[Dict[str, float], Dict[str, float], plt.Figure]:
    plain_p, plain_r = precision_recall_values(plain, get_iterator(dataset, [40]))
    cluster_p, cluster_r = precision_recall_values(with_clusters, get_iterator(dataset, [40]));

    # fig = plot({'With clusters': (cluster_r, cluster_p), 'Without clusters': (plain_r, plain_p)},
    #            'recall', 'precision')

    plain_scores = {'F1': max_f1(plain_p, plain_r),
                    'AoC': average_precision(plain_p, plain_r)}
    cluster_scores = {'F1': max_f1(cluster_p, cluster_r),
                    'AoC': average_precision(cluster_p, cluster_r)}
    return plain_scores, cluster_scores, None
