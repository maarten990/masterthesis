from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from sklearn.svm import SVC
from tabulate import tabulate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data import get_iterator, to_gpu, to_tensors
from train import setup_and_train

sns.set()


def get_values(model: nn.Module, dataloader: DataLoader, gpu: bool=True
               ) -> Tuple[List[float], List[float]]:
    """Get the classification output for the given dataset.

    :param model: A trained model.
    :param dataloader: The input data.
    :param gpu: If true, run on the gpu. Otherwise use the cpu.
    :returns: A list of classification values and a list of true samples.
    """
    model = model.eval()
    model.cuda() if gpu else model.cpu()

    predictions: List[float] = []
    true: List[float] = []

    for batch in dataloader:
        data = to_tensors(batch)
        if gpu:
            data = to_gpu(data)

        for _, d in data.items():
            X = d['data']
            c = d['cluster_data']
            y = d['label']

            pred = model(X, c)
            pred = pred.detach().cpu().squeeze(dim=1).numpy()
            predictions.extend(pred)
            true.extend(y.detach().cpu().squeeze(dim=1).numpy())

    return predictions, true


def evaluate_bow(model: SVC, vectorizer: TfidfVectorizer, dataset: Dataset,
                 cutoff: float = 0.5) -> Tuple[float, float]:
    samples = [list(entry.values())[0]['data'] for entry in dataset]
    true = [list(entry.values())[0]['label'] for entry in dataset]

    y = model.predict_proba(vectorizer.transform(samples))
    predictions = np.where(y > cutoff, 1, 0)
    predictions = np.argmax(predictions, axis=1)
    if 1 not in predictions:
        p = 1.0
    else:
        p = precision_score(true, predictions)
    r = recall_score(true, predictions)

    return p, r


def precision_recall_values(predicted: List[float], true: List[float]) -> Tuple[List[float], List[float]]:
    """Calculate the values for a  precision-recall curve.

    :param predicted: A list of classifier outputs.
    :param true: A list of the true classification labels.
    :returns: A list of (precision, recall) tuples, sorted by increasing recall.
    """
    p, r, _ = precision_recall_curve(true, predicted)
    return p, r


def average_precision(precision: List[float], recall: List[float]) -> float:
    return np.mean(precision)


def mean_aoc(precision: List[float], recall: List[float]) -> float:
    # m = np.mean(precision)
    # trapz = np.trapz(precision, recall)
    simp = scipy.integrate.simps(precision, recall)

    return simp


def mean_of_pr(precisions: List[List[float]], recalls: List[List[float]]) -> Dict[float, float]:
    buckets = np.linspace(0, 1)
    out: Dict[float, List[float]] = {b: [] for b in buckets}

    # iterate over trials
    for ps, rs in zip(precisions, recalls):
        for p, r in zip(ps, rs):
            for b, b_next in zip(buckets, buckets[1:]):
                if r < b_next:
                    out[b].append(p)
                    break

    for b in buckets:
        out[b] = np.mean(out[b]) if out[b] else 0.0

    # filter out any zero dips other than the last one, since they're just
    # caused by nothing happening to fall in that bin
    out = {b: v for b, v in out.items() if v != 0.0 or b == 1}

    return out


def max_f1(precision: List[float], recall: List[float]) -> float:
    p = np.array(precision)
    r = np.array(recall)
    return np.max(2 * ((p * r) / (p + r + 1e-3)))


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


def cross_val(k, train_size, model_fn, optim_fn, dataset, params, testset=None):
    folds = dataset.shuffle_split(k, train_size)
    F1s = []
    losses = []
    APs = []
    PRs = []

    test_on_holdout = testset is None

    for train, test in folds:
        torch.cuda.empty_cache()

        if test_on_holdout:
            testset = test

        model, loss = setup_and_train(params, model_fn, optim_fn, dataset=train,
                                      epochs=params.epochs, batch_size=50, gpu=True)
        losses.append(loss)
        scores = get_scores(model, testset)
        F1s.append(scores['F1'])
        APs.append(scores['AoC'])
        PRs.append(scores['pr'])

    return losses, PRs, F1s, APs


def analyze(data, filename_prefix=None):
    # filter NaNs
    # plain['F1'] = [x for x in plain['F1'] if not np.isnan(x)]
    # cluster['F1'] = [x for x in cluster['F1'] if not np.isnan(x)]
    # plain['AoC'] = [x for x in plain['AoC'] if not np.isnan(x)]
    # cluster['AoC'] = [x for x in cluster['AoC'] if not np.isnan(x)]

    # extract the items from the dict to guarantee a consistent iteration order
    items = list(data.items())

    print('Average convergence speed')
    loss_dict = {label: np.mean(losses, axis=0)
                 for label, (losses, _, _, _) in items}
    plot(loss_dict, 'epoch', 'loss')
    if filename_prefix:
        plt.savefig(f'{filename_prefix}_losses.pdf')
    plt.show()
    print()

    print('Average P/R curve')
    pr_dict = {}
    for label, (_, pr, _, _) in items:
        mean_pr = mean_of_pr([p for p, _ in pr],
                             [r for _, r in pr])
        r = sorted(mean_pr.keys())
        p = [mean_pr[r] for r in r]
        pr_dict[label] = (r, p)

    plot(pr_dict, 'recall', 'precision')
    if filename_prefix:
        plt.savefig(f'{filename_prefix}_pr.pdf')
    plt.show()
    print()

    print('Score table:')
    table = []
    for label, (_, pr, f1, aps) in items:
        mean_pr = mean_of_pr([p for p, _ in pr],
                             [r for _, r in pr])
        r = sorted(mean_pr.keys())
        p = [mean_pr[r] for r in r]
        table.append([label, np.mean(f1), np.std(f1), np.mean(aps), np.std(aps), mean_aoc(p, r)])

    print(tabulate(table, headers=['', 'F1 mean', 'F1 stddev', 'AoC mean',
                                   'AoC std', 'Area under averaged curve']))

    print()
    print('AP plots:')
    for label, (_, _, _, aps) in items:
        sns.distplot(aps, label=label)
    plt.legend()
    if filename_prefix:
        plt.savefig(f'{filename_prefix}_kde_ap.pdf')
    plt.show()

    df = pd.DataFrame({label: aps for label, (_, _, _, aps) in items})
    plt.figure()
    sns.boxplot(data=df)
    if filename_prefix:
        plt.savefig(f'{filename_prefix}_boxplot_ap.pdf')
    plt.show()

    print()
    print('F1 plots:')
    for label, (_, _, _, aps) in items:
        sns.distplot(aps, label=label)
    plt.legend()
    if filename_prefix:
        plt.savefig(f'{filename_prefix}_kde_f1.pdf')
    plt.show()

    df = pd.DataFrame({label: aps for label, (_, _, _, aps) in items})
    plt.figure()
    sns.boxplot(data=df)
    if filename_prefix:
        plt.savefig(f'{filename_prefix}_boxplot_f1.pdf')
    plt.show()

    print()
    print('Statistical significance:')
    sign_table_f1 = []
    sign_table_ap = []
    for label_1, (_, _, f1_1, ap_1) in items:
        sign_table_f1.append([label_1])
        sign_table_ap.append([label_1])
        for label_2, (_, _, f1_2, ap_2) in items:
            sign_table_f1[-1].append(scipy.stats.ttest_ind(f1_1, f1_2, equal_var=False).pvalue)
            sign_table_ap[-1].append(scipy.stats.ttest_ind(ap_1, ap_2, equal_var=False).pvalue)

    print('F1 score')
    print(tabulate(sign_table_f1, headers=[label for label, _ in items]))

    print()
    print('Area under curve')
    print(tabulate(sign_table_ap, headers=[label for label, _ in items]))
