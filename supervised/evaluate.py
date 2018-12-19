from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Iterator, Optional, Tuple, Union
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from tabulate import tabulate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import ClusterHandling, GermanDataset, get_iterator
from models import CategoricalClusterLabels, CNNClusterLabels, NoClusterLabels, OnlyClusterLabels
import train

sns.set()


def get_values(
        model: nn.Module,
        dataloader: DataLoader,
        gpu: bool = True,
        use_dist: bool = False,
        use_chars: bool = False,
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
        batch.to_tensors()
        if gpu:
            batch.to_gpu()

        X = batch.X_chars if use_chars else batch.X_words
        c = batch.clusters_gmm if use_dist else batch.clusters_kmeans
        y = batch.label

        pred = model(X, c)
        pred = pred.detach().cpu().view(-1).numpy()
        predictions.extend(pred)
        true.extend(y.detach().cpu().view(-1).numpy())
        batch.to_cpu()

    return predictions, true


def evaluate_bow(
        model: SVC, vectorizer: TfidfVectorizer, dataset: Dataset, vocab
) -> Tuple[float, float]:
    true = np.ravel([entry.label for entry in dataset])

    samples = [
        " ".join(
            [vocab.idx_to_token.get(idx, "NULL") for row in entry.X_words for idx in row]
        )
        for entry in dataset
    ]

    y = model.predict(vectorizer.transform(samples))
    return f1_score(true, y)


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
    wordlen: int,
    charlen: int,
    dataset: Dataset,
    gpu: bool = True,
    use_dist: bool = False,
    use_chars: bool = False,
) -> Dict[str, Any]:
    p, r = precision_recall_values(
        *get_values(
            model,
            get_iterator(dataset, wordlen, charlen)[0],
            gpu=gpu,
            use_dist=use_dist,
            use_chars=use_chars,
        )
    )

    scores = {"F1": max_f1(p, r), "AoC": average_precision(p, r), "pr": (p, r)}
    return scores


def shuffle_split(
        dataset: Dataset, k: int = 10, train_size: Union[float, int] = 0.9
) -> Iterator[Tuple[List[int], List[int]]]:
    """
    Return stratified training and testing folds with the same data
    distribution as the source data.
    :param k: The number of folds.
    :param train_size: If float, the proportion to use as training set. If
    int, the absolute number of samples to use.
    :returns: An iterator of (train, test) indices.
    """
    splitter = StratifiedShuffleSplit(
        n_splits=k, train_size=train_size, test_size=None, random_state=100
    )
    labels = dataset.get_labels()
    splits = splitter.split(np.zeros(len(dataset)), labels)

    for train_indices, test_indices in splits:
        yield train_indices, test_indices


def cross_val(
    k: int,
    splitter: StratifiedShuffleSplit,
    model_fns: List[Callable[[nn.Module], nn.Module]],
    use_dist_list: List[bool],
    optim_fn: Callable[[Any], torch.optim.Optimizer],
    dataset: Dataset,
    params: List[Any],
    early_stopping: int = 10,
    validation_set: Dataset = None,
    testset: Optional[Dataset] = None,
    gpu: bool = True,
    batch_size: int = 50,
):
    folds = splitter.split(np.zeros(len(dataset)), dataset.get_labels())

    F1s: List[List[float]] = [[] for _ in model_fns]
    losses: List[List[List[float]]] = [[] for _ in model_fns]
    APs: List[List[float]] = [[] for _ in model_fns]
    PRs: List[List[float]] = [[] for _ in model_fns]

    test_on_holdout = testset is None

    first_run = True
    for train_indices, test_indices in tqdm(folds, total=k, position=1):
        torch.cuda.empty_cache()

        for i, (model_fn, use_dist, parameters) in enumerate(
                zip(model_fns, use_dist_list, params)
        ):
            trainset, holdout = dataset.split_on(train_indices, test_indices)
            if test_on_holdout:
                testset = holdout

            if first_run:
                print(
                    f"{len(trainset)} training samples, {len(testset)} testing samples"
                )
                first_run = False

            model, loss, wordlen, charlen, use_chars = train.setup_and_train(
                parameters,
                model_fn,
                optim_fn,
                dataset=trainset,
                epochs=parameters.epochs,
                batch_size=batch_size,
                gpu=gpu,
                early_stopping=early_stopping,
                progbar=0,
                max_norm=parameters.max_norm,
                validation_set=validation_set,
                use_dist=use_dist,
            )
            scores = get_scores(
                model, wordlen, charlen, testset, gpu, use_dist, use_chars
            )
            losses[i].append(loss)
            F1s[i].append(scores["F1"])
            APs[i].append(scores["AoC"])
            PRs[i].append(scores["pr"])

    return losses, PRs, F1s, APs


def cross_val_bow(
        k: int,
        splitter: StratifiedShuffleSplit,
        dataset: Dataset,
        testset: Optional[Dataset] = None
) -> List[float]:
    folds = splitter.split(np.zeros(len(dataset)), dataset.get_labels())
    test_on_holdout = testset is None
    F1s: List[List[float]] = [[]]

    for train_indices, test_indices in tqdm(folds, total=k, position=1):
        trainset, holdout = dataset.split_on(train_indices, test_indices)
        if test_on_holdout:
            testset = holdout

        model, vectorizer = train.train_BoW(
            trainset, dataset.word_vocab, ngram_range=(1, 2)
        )
        F1s[0].append(
            evaluate_bow(model, vectorizer, testset, dataset.word_vocab)
        )

    return F1s, F1s, F1s, F1s


def analyze_wrapper(baseline, kmeans, gmm, variable="variable", path=None):
    for size in baseline.keys():
        data = {"baseline": baseline[size], "k-means": kmeans[size]}

        if gmm is not None:
            data["Clusters-LSTM"] = gmm[size]

        analyze(data, size, "model", path)


def analyze(data, size, variable="variable", path=None):
    # ensure the target folder exists
    if path and not os.path.isdir(path):
        os.makedirs(path)

    # extract the items from the dict to guarantee a consistent iteration order
    items = list(data.items())

    print("Average P/R curve")
    pr_dict = {}
    for label, (_, pr, _, _) in items:
        mean_pr = mean_of_pr([p for p, _ in pr], [r for _, r in pr])
        r = sorted(mean_pr.keys())
        p = [mean_pr[r] for r in r]
        pr_dict[label] = (r, p)

    plot(pr_dict, "recall", "precision")
    if path:
        plt.savefig(f"{path}/pr-{size}.pdf")
    plt.show()
    print()

    print("Score table:")
    table = []
    for label, (_, pr, f1, aps) in items:
        mean_pr = mean_of_pr([p for p, _ in pr], [r for _, r in pr])
        r = sorted(mean_pr.keys())
        p = [mean_pr[r] for r in r]
        table.append([label, np.mean(f1), np.std(f1)])

    print(tabulate(table, headers=[variable, "F1 mean", "F1 stddev"]))
    if path:
        with open(f"{path}/scores-{size}.tex", "w") as f:
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
        }
    )

    print()
    plt.figure()
    sns.boxplot(df[variable], df["F1 score"])
    if path:
        plt.savefig(f"{path}/boxplot_f1-{size}.pdf")
    plt.show()

    print()
    print("Statistical significance (dependent T-test):")
    sign_table_f1 = []
    for label_1, (_, _, f1_1, _) in items:
        sign_table_f1.append([label_1])
        for label_2, (_, _, f1_2, _) in items:
            sign_table_f1[-1].append(scipy.stats.ttest_rel(f1_1, f1_2, axis=0).pvalue)

    print("F1 score")
    print(tabulate(sign_table_f1, headers=[label for label, _ in items]))

    if path:
        with open(f"{path}/f1_sign-P{size}.tex", "w") as f:
            f.write(
                tabulate(
                    sign_table_f1,
                    headers=[label for label, _ in items],
                    tablefmt="latex_booktabs",
                )
            )


def analyze_size(
    data, ax="training samples", variable="variable", path=None, use_hue=False
):
    # ensure the target folder exists
    if path and not os.path.isdir(path):
        os.makedirs(path)

    df_data = {ax: [], variable: [], "F1 score": []}

    for size, d in data.items():
        items = list(d.items())
        df_data[ax] += [size for _, (_, _, f1, _) in items for _ in f1]
        df_data[variable] += [label for label, (_, _, f1, _) in items for _ in f1]
        df_data["F1 score"] += [score for _, (_, _, f1, _) in items for score in f1]

    df = pd.DataFrame(df_data)

    # overlay the plots in the same figure
    g = sns.factorplot(x=ax, y="F1 score", hue=variable, data=df, kind="point")
    g.set_titles("{col_name}")
    if path:
        plt.savefig(f"{path}/factorplot_f1_hue.pdf")
    plt.show()

    # display the plots side by side
    plt.figure()
    g = sns.factorplot(x=ax, y="F1 score", col=variable, data=df, kind="point")
    g.set_titles("{col_name}")
    plt.tight_layout()
    if path:
        plt.savefig(f"{path}/factorplot_f1_col.pdf")
    plt.show()


def analyze_tseries(data, ax="training samples", variable="variable", path=None):
    # ensure the target folder exists
    if path and not os.path.isdir(path):
        os.makedirs(path)

    df_data = {ax: [], variable: [], "F1 score": [], "trial": []}

    for size, d in data.items():
        items = list(d.items())
        df_data[ax] += [size for _, (_, _, f1, _) in items for _ in f1]
        df_data[variable] += [label for label, (_, _, f1, _) in items for _ in f1]
        df_data["F1 score"] += [score for _, (_, _, f1, _) in items for score in f1]
        df_data["trial"] += [i for _, (_, _, f1, _) in items for i, _ in enumerate(f1)]

    df = pd.DataFrame(df_data)

    # overlay the plots in the same figure
    sns.tsplot(
        time=ax,
        value="F1 score",
        condition=variable,
        unit="trial",
        data=df,
        err_style="ci_band",
        marker="o",
    )
    plt.tight_layout()
    if path:
        plt.savefig(f"{path}/tseries_f1.pdf")
    plt.show()


def analyze_cnns(data, ax="training samples", variable="variable", path=None):
    # ensure the target folder exists
    if path and not os.path.isdir(path):
        os.makedirs(path)

    df_data = {"architecture": [], ax: [], variable: [], "F1 score": [], "trial": []}

    for model, dset in data.items():
        for size, d in dset.items():
            items = list(d.items())
            df_data["architecture"] += [model for _, (_, _, f1, _) in items for _ in f1]
            df_data[ax] += [size for _, (_, _, f1, _) in items for _ in f1]
            df_data[variable] += [label for label, (_, _, f1, _) in items for _ in f1]
            df_data["F1 score"] += [score for _, (_, _, f1, _) in items for score in f1]
            df_data["trial"] += [
                i for _, (_, _, f1, _) in items for i, _ in enumerate(f1)
            ]

    df = pd.DataFrame(df_data)

    g = sns.relplot(
        x=ax,
        y="F1 score",
        hue=variable,
        col="architecture",
        data=df,
        kind="line",
        marker="o",
        ci=90,
    )
    g.set_titles("{col_name}")
    if path:
        plt.savefig(f"{path}/tseries_f1.pdf")
    plt.show()


def load_dataset(
    folder: str,
    gmm_folder: str,
    num_clusters: int,
    gmm_clusters: int,
    num_before: int = 0,
    num_after: int = 1,
    old_test: bool = False,
    bag_of_words: bool = False,
    cluster_handling: ClusterHandling = ClusterHandling.CONCAT,
) -> Tuple[Dataset, Dataset, Dataset]:
    files = [f"{folder}/18{i:03d}.xml" for i in [1, 2, 3, 4, 5, 6, 7, 209, 210, 211]]
    test_files = [f"{folder}/{i}162.xml" for i in [14, 15, 16]]
    valid_files = [f"{folder}/{i}019.xml" for i in [14, 15, 16, 17]]
    all_files = files + valid_files + test_files

    files_gmm = [f"{gmm_folder}/18{i:03d}.xml" for i in [1, 2, 3, 4, 5, 6, 7, 209, 210, 211]]
    test_files_gmm = [f"{gmm_folder}/{i}162.xml" for i in [14, 15, 16]]
    valid_files_gmm = [f"{gmm_folder}/{i}019.xml" for i in [14, 15, 16, 17]]
    all_files_gmm = files_gmm + valid_files_gmm + test_files_gmm

    vocab_set = GermanDataset(
        all_files,
        all_files_gmm,
        num_clusters,
        gmm_clusters,
        num_before,
        num_after,
        bag_of_words=bag_of_words,
        cluster_handling=cluster_handling,
    )
    word_vocab = vocab_set.word_vocab
    char_vocab = vocab_set.char_vocab
    del vocab_set

    validset = GermanDataset(
        valid_files,
        valid_files_gmm,
        num_clusters,
        gmm_clusters,
        num_before,
        num_after,
        word_vocab=word_vocab,
        char_vocab=char_vocab,
        bag_of_words=bag_of_words,
        cluster_handling=cluster_handling,
    )

    if old_test:
        dataset = GermanDataset(
            files,
            files_gmm,
            num_clusters,
            gmm_clusters,
            num_before,
            num_after,
            word_vocab=word_vocab,
            char_vocab=char_vocab,
            bag_of_words=bag_of_words,
            cluster_handling=cluster_handling,
        )
        testset = GermanDataset(
            test_files,
            test_files_gmm,
            num_clusters,
            gmm_clusters,
            num_before,
            num_after,
            word_vocab=word_vocab,
            char_vocab=char_vocab,
            bag_of_words=bag_of_words,
            cluster_handling=cluster_handling,
        )

        retval = (dataset, validset, testset)
    else:
        dataset = GermanDataset(
            files + test_files,
            files_gmm + test_files_gmm,
            num_clusters,
            gmm_clusters,
            num_before,
            num_after,
            word_vocab=word_vocab,
            char_vocab=char_vocab,
            bag_of_words=bag_of_words,
            cluster_handling=cluster_handling,
        )
        retval = (dataset, validset)

    return retval


Results = namedtuple("Results", ["baseline", "dbscan", "gmm"])


def run(
    word_params: Optional["CNNParams"],
    char_params: Optional["CNNParams"],
    training_sizes: List[int],
    window_sizes: List[Tuple[int, int]],
    k: int = 5,
    nocluster_dropout: float = 0.5,
    kmeans_path: str = "../clustered",
    gmm_path: str = "../clustered_gmm",
    num_clusters: int = 10,
    num_clusters_gmm: int = 10,
    use_cluster_cnn: bool = False,
    use_only_clusters: bool = False,
    use_bow: bool = False,
) -> Tuple[Results, Results]:
    if not (word_params or char_params):
        print("Need at least one of {word_params, char_params")
        return Results(None, None, None), Results(None, None, None)

    both_models = word_params and char_params

    baseline = defaultdict(dict)
    dbscan = defaultdict(dict)
    gmm = defaultdict(dict)
    char_baseline = defaultdict(dict)
    char_dbscan = defaultdict(dict)
    char_gmm = defaultdict(dict)

    if use_cluster_cnn:
        def fn(w, n):
            return lambda r: CNNClusterLabels(r, w, n, word_params.dropout)
    elif use_only_clusters:
        def fn(w, n):
            return lambda r: OnlyClusterLabels(r, n * (sum(w) + 1), word_params.dropout)
    else:
        def fn(w, n):
            return lambda r: CategoricalClusterLabels(r, n * (sum(w) + 1), word_params.dropout)

    for training_size in training_sizes:
        for window_size in window_sizes:
            optim_fn = lambda p: torch.optim.Adam(p)
            model_fns = []

            if nocluster_dropout >= 0:
                model_fns.append(lambda r: NoClusterLabels(r, nocluster_dropout))
            if word_params:
                model_fns += [
                    fn(window_size, num_clusters),
                    fn(window_size, num_clusters_gmm),
                ]

            if nocluster_dropout >= 0:
                model_fns.append(lambda r: NoClusterLabels(r, nocluster_dropout))
            if char_params:
                model_fns += [
                    fn(window_size, num_clusters),
                    fn(window_size, num_clusters_gmm),
                ]

            dataset, validset, testset = load_dataset(
                kmeans_path, gmm_path, num_clusters, num_clusters_gmm, window_size[0], window_size[1], old_test=True
            )
            splitter = StratifiedShuffleSplit(
                n_splits=k,
                train_size=training_size,
                test_size=None,
                random_state=100,
            )

            params_list = []
            multiplier = 3 if nocluster_dropout >= 0 else 2
            params_list += ([word_params] * multiplier) if word_params else []
            params_list += ([char_params] * multiplier) if char_params else []

            use_dist_list: List[bool]
            if nocluster_dropout >= 0:
                use_dist_list = [False, False, True] * (2 if both_models else 1)
            else:
                use_dist_list = [False, True] * (2 if both_models else 1)

            splitter.random_state = 100
            if use_bow:
                values = cross_val_bow(k, splitter, dataset, testset=testset)
            else:
                values = cross_val(
                    k,
                    splitter,
                    model_fns,
                    use_dist_list,
                    optim_fn,
                    dataset,
                    params=params_list,
                    early_stopping=3,
                    validation_set=validset,
                    batch_size=128,
                    testset=testset,
                )

            result_order = []
            if word_params:
                if nocluster_dropout >= 0:
                    result_order.append(baseline)
                result_order += [dbscan, gmm]
            if char_params:
                if nocluster_dropout >= 0:
                    result_order.append(char_baseline)
                result_order += [char_dbscan, char_gmm]

            if use_bow:
                # special case, override the order
                result_order = [baseline]

            num_iter = len(values[0])
            assert(num_iter == len(result_order))

            for i, var in enumerate(result_order):
                var[window_size][training_size] = [v[i] for v in values]

    return (
        Results(baseline, dbscan, gmm),
        Results(char_baseline, char_dbscan, char_gmm),
    )


def results_to_dataframe(
        word_results: Results, char_results: Results
) -> pd.DataFrame:
    table = []
    for win in word_results.baseline.keys():
        for train_size in word_results.baseline[win].keys():
            for i in range(len(word_results.baseline[win][train_size][2])):
                table.extend(
                    [
                        ["TokenCNN", "Baseline", sum(win), train_size, word_results.baseline[win][train_size][2][i]],
                        ["TokenCNN", "K-Means", sum(win), train_size, word_results.dbscan[win][train_size][2][i]],
                        ["TokenCNN", "GMM", sum(win), train_size, word_results.gmm[win][train_size][2][i]],
                        ["CharCNN", "Baseline", sum(win), train_size, char_results.baseline[win][train_size][2][i]],
                        ["CharCNN", "K-Means", sum(win), train_size, char_results.dbscan[win][train_size][2][i]],
                        ["CharCNN", "GMM", sum(win), train_size, char_results.gmm[win][train_size][2][i]],
                    ]
                )

    df = pd.DataFrame.from_records(
        table, columns=["model", "method", "window", "size", "score"]
    )

    return df


def plot_sns(df: pd.DataFrame) -> None:
    g = sns.catplot(
        x="size", y="score", data=df, kind="box", hue="method", col="model", row="window"
    )
    g.set_axis_labels("Number of training samples", "F1 score")
    g.set_titles("{col_name}, window size {row_name}")
