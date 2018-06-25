from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os

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
from tqdm import tqdm

from data import get_iterator, to_gpu, to_tensors
import train

sns.set()


def get_values(
    model: nn.Module, dataloader: DataLoader, gpu: bool = True, idx_only: bool = False
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

    cluster_str = "cluster_data_only_idx" if idx_only else "cluster_data_full"

    for batch in dataloader:
        data = to_tensors(batch)
        if gpu:
            data = to_gpu(data)

        for _, d in data.items():
            X = d["data"]
            c = d[cluster_str]
            y = d["label"]

            pred = model(X, c)
            pred = pred.detach().cpu().squeeze(dim=1).numpy()
            predictions.extend(pred)
            true.extend(y.detach().cpu().squeeze(dim=1).numpy())

    return predictions, true


def evaluate_bow(
    model: SVC, vectorizer: TfidfVectorizer, dataset: Dataset, cutoff: float = 0.5
) -> Tuple[float, float]:
    samples = [list(entry.values())[0]["data"] for entry in dataset]
    true = [list(entry.values())[0]["label"] for entry in dataset]

    y = model.predict_proba(vectorizer.transform(samples))
    predictions = np.where(y > cutoff, 1, 0)
    predictions = np.argmax(predictions, axis=1)
    if 1 not in predictions:
        p = 1.0
    else:
        p = precision_score(true, predictions)
    r = recall_score(true, predictions)

    return p, r


def precision_recall_values(
    predicted: List[float], true: List[float]
) -> Tuple[List[float], List[float]]:
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


def mean_of_pr(
    precisions: List[List[float]], recalls: List[List[float]]
) -> Dict[float, float]:
    buckets = np.linspace(0, 1)
    hist: Dict[float, List[float]] = {b: [] for b in buckets}

    # iterate over trials
    for ps, rs in zip(precisions, recalls):
        for p, r in zip(ps, rs):
            for b, b_next in zip(buckets, buckets[1:]):
                if r < b_next:
                    hist[b].append(p)
                    break

    out: Dict[float, float] = {
        b: np.mean(values) if values else 0.0 for b, values in hist.items()
    }

    # filter out any zero dips other than the last one, since they're just
    # caused by nothing happening to fall in that bin
    return {b: v for b, v in out.items() if v != 0.0 or b == 1}


def max_f1(precision: List[float], recall: List[float]) -> float:
    p = np.array(precision)
    r = np.array(recall)
    return np.max(2 * ((p * r) / (p + r + 1e-3)))


def plot(
    curves: Dict[str, Union[List[float], Tuple[List[float], List[float]]]],
    xlabel: str,
    ylabel: str,
    title: str = "",
) -> plt.Figure:
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


def get_scores(
    model: nn.Module,
    buckets: List[int],
    dataset: Dataset,
    gpu: bool = True,
    idx_only: bool = False,
) -> Dict[str, Any]:
    p, r = precision_recall_values(
        *get_values(
            model, get_iterator(dataset, buckets)[0], gpu=gpu, idx_only=idx_only
        )
    )

    scores = {"F1": max_f1(p, r), "AoC": average_precision(p, r), "pr": (p, r)}
    return scores


def cross_val(
    k: int,
    train_size: int,
    model_fns: List[Callable[[nn.Module], nn.Module]],
    idx_only_list: List[bool],
    optim_fn: Callable[[Any], torch.optim.Optimizer],
    dataset: Dataset,
    params: Any,
    early_stopping: int = 10,
    validation_set: Dataset = None,
    testset: Optional[Dataset] = None,
    gpu: bool = True,
):
    if train_size == -1:
        folds = dataset.kfold(k)
    else:
        folds = dataset.shuffle_split(k, train_size)

    F1s: List[List[float]] = [[] for _ in model_fns]
    losses: List[List[List[float]]] = [[] for _ in model_fns]
    APs: List[List[float]] = [[] for _ in model_fns]
    PRs: List[List[float]] = [[] for _ in model_fns]

    test_on_holdout = testset is None

    first_run = True
    for trainset, test in tqdm(folds, total=k):
        torch.cuda.empty_cache()

        if test_on_holdout:
            testset = test

        if first_run:
            print(f"{len(trainset)} training samples, {len(testset)} testing samples")
            first_run = False

        for i, (model_fn, idx_only) in enumerate(zip(model_fns, idx_only_list)):
            model, loss, buckets = train.setup_and_train(
                params,
                model_fn,
                optim_fn,
                dataset=trainset,
                epochs=params.epochs,
                batch_size=50,
                gpu=gpu,
                early_stopping=early_stopping,
                progbar=False,
                max_norm=params.max_norm,
                validation_set=validation_set,
                idx_only=idx_only,
            )
            scores = get_scores(model, buckets, testset, gpu)
            losses[i].append(loss)
            F1s[i].append(scores["F1"])
            APs[i].append(scores["AoC"])
            PRs[i].append(scores["pr"])

    return losses, PRs, F1s, APs


def analyze(data, variable="variable", path=None):
    # ensure the target folder exists
    if path and not os.path.isdir(path):
        os.makedirs(path)

    # extract the items from the dict to guarantee a consistent iteration order
    items = list(data.items())

    print("Average convergence speed")
    max_epoch = np.max(
        [len(sample) for _, (losses, _, _, _) in items for sample in losses]
    )
    loss_dict = {
        label: np.mean(
            [
                np.pad(sample, ((0, (max_epoch - len(sample)))), "edge")
                for sample in losses
            ],
            axis=0,
        )
        for label, (losses, _, _, _) in items
    }
    plot(loss_dict, "epoch", "loss")
    if path:
        plt.savefig(f"{path}/losses.pdf")
    plt.show()
    print()

    print("Average P/R curve")
    pr_dict = {}
    for label, (_, pr, _, _) in items:
        mean_pr = mean_of_pr([p for p, _ in pr], [r for _, r in pr])
        r = sorted(mean_pr.keys())
        p = [mean_pr[r] for r in r]
        pr_dict[label] = (r, p)

    plot(pr_dict, "recall", "precision")
    if path:
        plt.savefig(f"{path}/pr.pdf")
    plt.show()
    print()

    print("Score table:")
    table = []
    for label, (_, pr, f1, aps) in items:
        mean_pr = mean_of_pr([p for p, _ in pr], [r for _, r in pr])
        r = sorted(mean_pr.keys())
        p = [mean_pr[r] for r in r]
        table.append(
            [label, np.mean(f1), np.std(f1), np.mean(aps), np.std(aps), mean_aoc(p, r)]
        )

    print(
        tabulate(
            table,
            headers=[
                variable,
                "F1 mean",
                "F1 stddev",
                "AUC mean",
                "AUC std",
                "Area under averaged curve",
            ],
        )
    )
    if path:
        with open(f"{path}/scores.tex", "w") as f:
            f.write(
                tabulate(
                    table,
                    headers=[
                        variable,
                        "F1 mean",
                        "F1 stddev",
                        "AUC mean",
                        "AUC std",
                        "Area under averaged curve",
                    ],
                    tablefmt="latex_booktabs",
                )
            )

    df = pd.DataFrame(
        {
            variable: [label for label, (_, _, f1, _) in items for _ in f1],
            "F1 score": [score for _, (_, _, f1, _) in items for score in f1],
            "Area under curve": [score for _, (_, _, _, aoc) in items for score in aoc],
        }
    )

    print()
    print("F1 plots:")
    for label, (_, _, f1, _) in items:
        sns.distplot(df[df[variable] == label]["F1 score"], hist=False, label=label)
    plt.legend()
    if path:
        plt.savefig(f"{path}/kde_f1.pdf")
    plt.show()

    plt.figure()
    sns.boxplot(df[variable], df["F1 score"])
    if path:
        plt.savefig(f"{path}/boxplot_f1.pdf")
    plt.show()

    print()
    print("Statistical significance (dependent T-test):")
    sign_table_f1 = []
    sign_table_ap = []
    for label_1, (_, _, f1_1, ap_1) in items:
        sign_table_f1.append([label_1])
        sign_table_ap.append([label_1])
        for label_2, (_, _, f1_2, ap_2) in items:
            sign_table_f1[-1].append(scipy.stats.ttest_rel(f1_1, f1_2, axis=0).pvalue)
            sign_table_ap[-1].append(scipy.stats.ttest_rel(ap_1, ap_2, axis=0).pvalue)

    print("F1 score")
    print(tabulate(sign_table_f1, headers=[label for label, _ in items]))

    if path:
        with open(f"{path}/f1_sign.tex", "w") as f:
            f.write(
                tabulate(
                    sign_table_f1,
                    headers=[label for label, _ in items],
                    tablefmt="latex_booktabs",
                )
            )


def analyze_size(data, variable="variable", path=None):
    # ensure the target folder exists
    if path and not os.path.isdir(path):
        os.makedirs(path)

    df_data = {"training samples": [], variable: [], "F1 score": []}

    for size, d in data.items():
        items = list(d.items())
        df_data["training samples"] += [size for _, (_, _, f1, _) in items for _ in f1]
        df_data[variable] += [label for label, (_, _, f1, _) in items for _ in f1]
        df_data["F1 score"] += [score for _, (_, _, f1, _) in items for score in f1]

    df = pd.DataFrame(df_data)

    g = sns.factorplot(
        x="training samples", y="F1 score", col=variable, data=df, kind="point"
    )
    g.set_titles("{col_name}")
    if path:
        plt.savefig(f"{path}/factorplot_f1.pdf")
    plt.show()
