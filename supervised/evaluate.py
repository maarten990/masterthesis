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


def get_values(model: nn.Module, dataloader: DataLoader, gpu: bool=True
              ) -> Tuple[List[float], List[bool]]:
    """Get the classification output for the given dataset.

    :param model: A trained model.
    :param dataloader: The input data.
    :param gpu: If true, run on the gpu. Otherwise use the cpu.
    :returns: A list of classification values and a list of true samples.
    """
    model = model.eval()
    model.cuda() if gpu else model.cpu()

    predictions: List[float] = []
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
            predictions.extend(pred)
            true.extend(y.data.cpu().numpy())

    return predictions, true


def evaluate_clf():
    pass


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


def precision_recall_values(predicted: List[float], true: List[bool]) -> Tuple[List[float], List[float]]:
    """Calculate the values for a  precision-recall curve.

    :param predicted: A list of classifier outputs.
    :param true: A list of the true classification labels.
    :returns: A list of (precision, recall) tuples, sorted by increasing recall.
    """
    pr: List[Tuple[float, float]] = []

    # a list of indices into the predicted/true lists sorted by classification value
    indices = sorted(list(range(len(predicted))), key=lambda i: predicted[i], reverse=True)
    total_positives = [true[idx] for idx in indices].count(1)
    previous_positives = 0
    for i in range(1, len(indices)):
        idxs = indices[:i]
        classifications = [true[idx] for idx in idxs]
        positives = classifications.count(1)
        if positives > previous_positives:
            previous_positives = positives
            pr.append((positives / len(classifications), positives / total_positives))

        if positives == total_positives:
            break

    # split the tuples into the 2 lists to return
    p, r = np.array(pr).T
    return p, r


def average_precision(precision: List[float], recall: List[float]) -> float:
    return np.mean(precision)


def mean_of_pr(precisions, recalls):
    buckets = np.linspace(0, 1)
    out = {b: [] for b in buckets}

    # iterate over trials
    for ps, rs in zip(precisions, recalls):
        for p, r in zip(ps, rs):
            for b in buckets:
                if r >= b:
                    out[b].append(p)

    for b in buckets:
        out[b] = np.mean(out[b])

    return out


def max_f1(precision: List[float], recall: List[float]) -> float:
    p = np.array(precision)
    r = np.array(recall)
    return np.max(2 * ((p * r) / (p + r)))


def plot(curves: Dict[str, Union[List[float], Tuple[List[float], List[float]]]], xlabel: str, ylabel: str,
         title: str = '') -> plt.Figure:
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
            sorted_indices = sorted(list(range(len(x))), key=lambda i: x[i])
            x = [x[i] for i in sorted_indices]
            y = [y[i] for i in sorted_indices]

        else:
            x = list(range(len(values)))
            y = values

        ax.plot(x, y, label=label)

    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    return ax


def get_scores(model: nn.Module, dataset: Dataset) -> Dict[str, float]:
    p, r = precision_recall_values(*get_values(model, get_iterator(dataset, [40])))

    scores = {'F1': max_f1(p, r),
              'AoC': average_precision(p, r),
              'pr': (p, r)}
    return scores
