"""Contains functions for loading and manipulating training data."""


from enum import Enum
import string
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

from lxml import etree
import nltk
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm

charmap = string.ascii_letters + string.digits + string.punctuation


def full_window_dist_fn(
        window,
        num_clusterlabels,
        center=None,
        func=lambda x: np.mean(x, axis = 0),
):
    if center is not None:
        return (
            eval(center.attrib["clusterLabel"])
            if "clusterLabel" in center.attrib
            else ([0.0] * num_clusterlabels)
        )
    else:
        return func(
            [
                eval(w.attrib["clusterLabel"])
                if "clusterLabel" in w.attrib
                else ([0.0] * num_clusterlabels)
                for w in window
            ],
        )


class Vocab:

    def __init__(
        self, token_to_idx: Dict[str, int], idx_to_token: Dict[int, str]
    ) -> None:
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token


class Sample:

    def __init__(self,
                 X_words,
                 X_chars,
                 clusters_kmeans,
                 clusters_gmm,
                 label,
                 bag_of_words: bool = False) -> None:

        if bag_of_words:
            self.X_words = X_words
            self.X_chars = X_chars
        else:
            self.X_words = self.ensure_2d(X_words)
            self.X_chars = self.ensure_2d(X_chars)
        self.clusters_kmeans = self.ensure_2d(clusters_kmeans)
        self.clusters_gmm = self.ensure_2d(clusters_gmm)
        self.label = label

    def __repr__(self) -> str:
        return f"""Sample(
    {self.X_words.shape}
    {self.X_chars.shape}
    {self.clusters_kmeans.shape}
    {self.clusters_gmm.shape}
    {self.label.shape}
)"""

    def __eq__(self, other: "Sample") -> str:
        return (
            np.all(self.X_words == other.X_words)
            and np.all(self.X_chars == other.X_chars)
            and np.all(self.label == other.label)
        )

    def __add__(self, other: "Sample") -> "Sample":
        return Sample(
            np.append(self.X_words, other.X_words, 0),
            np.append(self.X_chars, other.X_chars, 0),
            np.append(self.clusters_kmeans, other.clusters_kmeans, 0),
            np.append(self.clusters_gmm, other.clusters_gmm, 0),
            np.append(self.label, other.label, 0),
        )

    def ensure_2d(self, arr: np.ndarray) -> np.ndarray:
        return np.expand_dims(arr, 0) if len(arr.shape) < 2 else arr

    def to_tensors(self) -> "Sample":
        self.X_words = Variable(torch.from_numpy(self.X_words)).long()
        self.X_chars = Variable(torch.from_numpy(self.X_chars)).long()
        self.clusters_kmeans = Variable(torch.from_numpy(self.clusters_kmeans)).long()
        self.clusters_gmm = Variable(torch.from_numpy(self.clusters_gmm)).float()
        self.label = Variable(torch.from_numpy(self.label)).float()

        return self

    def to_gpu(self) -> "Sample":
        if torch.cuda.is_available():
            self.X_words = self.X_words.cuda()
            self.X_chars = self.X_chars.cuda()
            self.clusters_kmeans = self.clusters_kmeans.cuda()
            self.clusters_gmm = self.clusters_gmm.cuda()
            self.label = self.label.cuda()

        return self

    def to_cpu(self) -> "Sample":
        self.X_words = self.X_words.cpu()
        self.X_chars = self.X_chars.cpu()
        self.clusters_kmeans = self.clusters_kmeans.cpu()
        self.clusters_gmm = self.clusters_gmm.cpu()
        self.label = self.label.cpu()

        return self

    def pad(self, word_amount: int, char_amount: int) -> None:
        if word_amount >= 0:
            self.X_words = np.pad(self.X_words, [(0, 0), (0, word_amount)], "constant")
        else:
            self.X_words = self.X_words[:, :word_amount]

        if char_amount >= 0:
            self.X_chars = np.pad(self.X_chars, [(0, 0), (0, char_amount)], "constant")
        else:
            self.X_chars = self.X_chars[:, :char_amount]


ClusterHandling = Enum("ClusterHandling", "CONCAT MEAN CENTER")


class GermanDataset(Dataset):

    def __init__(
        self,
        files: List[str],
        gmm_files: List[str],
        num_clusterlabels: int,
        num_gmm_clusters: int,
        window_before: int,
        window_after: int,
        word_vocab: Optional[Vocab] = None,
        char_vocab: Optional[Vocab] = None,
        bag_of_words: bool = False,
        cluster_handling: ClusterHandling = ClusterHandling.CONCAT
    ) -> None:
        self.word_vocab = (
            create_dictionary(files, False) if not word_vocab else word_vocab
        )
        self.char_vocab = (
            create_dictionary(files, True) if not char_vocab else char_vocab
        )
        self.num_clusterlabels = num_clusterlabels
        self.num_gmm_clusters = num_gmm_clusters
        self.window_before = window_before
        self.window_after = window_after
        self.samples: List[Sample] = []
        self.bag_of_words = bag_of_words
        self.rng = np.random.RandomState()
        self.cluster_handling = cluster_handling

        for file, gmm_file in zip(files, gmm_files):
            xml = load_xml_from_disk(file)
            xml_gmm = load_xml_from_disk(gmm_file)
            flatten_xml(xml)
            flatten_xml(xml_gmm)

            pos = xml.xpath('/pdf2xml/text[@is-speech="true"]')
            neg = xml.xpath('/pdf2xml/text[@is-speech="false"]')[10:-10]
            pos_gmm = xml_gmm.xpath('/pdf2xml/text[@is-speech="true"]')
            neg_gmm = xml_gmm.xpath('/pdf2xml/text[@is-speech="false"]')[10:-10]

            # take a random amount of negative samples equal to the number of
            # positive samples
            self.rng.seed(100)
            indices = self.rng.choice(list(range(len(neg))), len(pos), replace=False)
            neg = [neg[i] for i in indices]
            neg_gmm = [neg_gmm[i] for i in indices]

            for p, p_gmm in zip(pos, pos_gmm):
                assert(p.xpath(".//text()") == p_gmm.xpath(".//text()"))
                window = xml_window(p, window_before, window_after)
                window_gmm = xml_window(p_gmm, window_before, window_after)
                if window:
                    self.samples.append(self.vectorize_window(window, window_gmm, p, p_gmm))
            for n, n_gmm in zip(neg, neg_gmm):
                assert(n.xpath(".//text()") == n_gmm.xpath(".//text()"))
                window = xml_window(n, window_before, window_after)
                window_gmm = xml_window(n_gmm, window_before, window_after)
                if window:
                    self.samples.append(self.vectorize_window(window, window_gmm, n, n_gmm))

    def get_pos_neg(self) -> Tuple[List[int], List[int]]:
        positives: List[int] = []
        negatives: List[int] = []
        for i, sample in enumerate(self.samples):
            if (sample.label == 1).all():
                positives.append(i)
            else:
                negatives.append(i)

        return positives, negatives

    def get_labels(self) -> List[int]:
        """
        Return the labels belonging to each sample.
        """
        labels: List[int] = []
        for i, sample in enumerate(self.samples):
            labels.append(sample.label)

        return labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def split(
        self, train: Tuple[int, int], test: Optional[Tuple[int, int]] = None
    ) -> Tuple[Dataset, Dataset]:
        """
        Split the dataset into a training set and a test set.
        :param train: A tuple indicating the number of positive and negative
            samples in the training set.
        :param test: An optional tuple indicating the number of positive and
            negative samples in the test set.
        :returns: A tuple of two new datasets.
        """
        if not test:
            return_test = False
            test = (0, 0)
        else:
            return_test = True

        positives, negatives = self.get_pos_neg()
        self.rng.seed(100)
        pos_indices = self.rng.choice(positives, train[0] + test[0], replace=False)
        neg_indices = self.rng.choice(negatives, train[1] + test[1], replace=False)

        train_indices = np.concatenate(
            (pos_indices[: train[0]], neg_indices[: train[0]])
        )
        test_indices = np.concatenate((pos_indices[: test[0]], neg_indices[: test[0]]))
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        if return_test:
            return DataSubset(self, train_indices), DataSubset(self, test_indices)
        else:
            return DataSubset(self, train_indices)

    def kfold(self, k: int = 10) -> Iterator[Tuple[Dataset, Dataset]]:
        """
        Return stratified training and testing folds with the same data
        distribution as the source data.
        :param k: The number of folds.
        :returns: An iterator of (train, test) datasets.
        """
        fold = StratifiedKFold(n_splits=k, shuffle=True)
        labels = self.get_labels()
        split = fold.split(np.zeros(len(self)), labels)

        for train_indices, test_indices in split:
            yield DataSubset(self, train_indices), DataSubset(self, test_indices)

    def shuffle_split(
        self, k: int = 10, train_size: Union[float, int] = 0.9
    ) -> Iterator[Tuple[Dataset, Dataset]]:
        """
        Return stratified training and testing folds with the same data
        distribution as the source data.
        :param k: The number of folds.
        :param train_size: If float, the proportion to use as training set. If
            int, the absolute number of samples to use.
        :returns: An iterator of (train, test) datasets.
        """
        splitter = StratifiedShuffleSplit(
            n_splits=k, train_size=train_size, test_size=None
        )
        labels = self.get_labels()
        splits = splitter.split(np.zeros(len(self)), labels)

        for train_indices, test_indices in splits:
            yield DataSubset(self, train_indices), DataSubset(self, test_indices)

    def split_on(
        self, train_indices: List[int], test_indices: List[int]
    ) -> Tuple[Dataset, Dataset]:
        return DataSubset(self, train_indices), DataSubset(self, test_indices)

    def vectorize_window(
        self,
        window: List[etree._Element],
        window_gmm: List[etree._Element],
        center_kmeans: etree._Element,
        center_gmm: etree._Element,
    ) -> Sample:
        tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+|[^\w\s]")
        y = get_label(center_kmeans)

        word_tokens = token_featurizer(window, tokenizer)
        char_tokens = char_tokenizer(window, tokenizer)

        if self.bag_of_words:
            X_words = " ".join(word_tokens)
            X_chars = " ".join(char_tokens)
        else:
            X_words = np.array(
                [self.word_vocab.token_to_idx.get(token, 0) for token in word_tokens]
            )
            X_chars = np.array(
                [self.char_vocab.token_to_idx.get(token, 0) for token in char_tokens]
            )

        use_center_kmeans, use_center_gmm = {
            ClusterHandling.CENTER: (center_kmeans, center_gmm),
        }.get(self.cluster_handling, (None, None))

        cluster_fn = {
            ClusterHandling.CENTER: None,
            ClusterHandling.CONCAT: lambda x: np.concatenate(x),
            ClusterHandling.MEAN: lambda x: np.mean(x, axis=0),
        }[self.cluster_handling]

        clusterlabels = full_window_dist_fn(
            window, self.num_clusterlabels, use_center_kmeans, cluster_fn
        )
        clusterlabels_gmm = full_window_dist_fn(
            window_gmm, self.num_gmm_clusters, use_center_gmm, cluster_fn
        )

        return Sample(
            X_words,
            X_chars,
            np.array(clusterlabels),
            np.array(clusterlabels_gmm),
            np.array([y]),
            bag_of_words=self.bag_of_words,
        )


class DataSubset(GermanDataset):

    def __init__(self, data: GermanDataset, indices: List[int]) -> None:
        self.samples = [data.samples[i] for i in indices]
        self.word_vocab = data.word_vocab
        self.char_vocab = data.char_vocab
        self.num_clusterlabels = data.num_clusterlabels


def xml_window(node: etree._Element, n_before: int, n_after: int) -> List[etree._Element]:
    start = node
    for _ in range(n_before):
        prev = start.getprevious()
        if prev is not None:
            start = prev
        else:
            print("Reached boundary!")
            return []

    out = []
    for _ in range(n_before + n_after + 1):
        if start is None:
            print("Reached boundary!")
            return []
        out.append(start)
        start = start.getnext()

    return out


def get_iterator(
    dataset: Dataset,
    wordlen: Optional[int],
    charlen: Optional[int],
    batch_size: int = 32,
) -> Tuple[DataLoader, int, int]:
    if not wordlen:
        # find the bucket size that fits 90% of the data
        sizes = [np.size(sample.X_words) for sample in dataset]
        cutoff = int(np.percentile(sizes, 90))
        wordlen = cutoff
    if not charlen:
        # find the bucket size that fits 90% of the data
        sizes = [np.size(sample.X_chars) for sample in dataset]
        cutoff = int(np.percentile(sizes, 90))
        charlen = cutoff

    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=CollateWithSeqlen(wordlen, charlen),
        ),
        wordlen,
        charlen,
    )


class CollateWithSeqlen:

    def __init__(self, word_len: int, char_len: int) -> None:
        self.word_len = word_len
        self.char_len = char_len

    def __call__(self, samples: List[Sample]) -> Sample:
        return self.bucket(samples)

    def bucket(self, samples: List[Sample]) -> Sample:
        out: Sample = None
        for s in samples:
            s.pad(
                self.word_len - np.size(s.X_words), self.char_len - np.size(s.X_chars)
            )
            out = (out + (s)) if out else s

        return out


def load_xml_from_disk(path: str) -> etree._Element:
    parser = etree.XMLParser(ns_clean=True, encoding="utf-8")
    with open(path, "r", encoding="utf-8") as f:
        return etree.fromstring(f.read().encode("utf-8"), parser)


def flatten_xml(xml: etree._Element):
    pages = xml.xpath('/pdf2xml/page')
    children = []
    for page in pages:
        for child in page:
            children.append(child)
        xml.remove(page)

    for child in children:
        xml.append(child)


def get_label(node: etree._Element) -> int:
    is_speech = node.attrib["is-speech"]
    return 1 if is_speech == "true" else 0


def get_speaker(node: etree._Element) -> str:
    is_speech = node.attrib["is-speech"]

    if is_speech == "true":
        return node.attrib["speaker"]
    else:
        return ""


def create_dictionary(paths: List[str], char_tokens=False) -> Vocab:
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+|[^\w\s]")
    all_words: Set[str] = set()

    for path in tqdm(paths, desc="Creating dictionary"):
        xml = load_xml_from_disk(path)
        text = " ".join(xml.xpath("/pdf2xml/page/text/text()")).lower()
        if char_tokens:
            tokens = [char for token in tokenizer.tokenize(text) for char in token]
        else:
            tokens = tokenizer.tokenize(text)
        all_words |= set(tokens)

    word_to_idx = {w: i + 1 for i, w in enumerate(all_words)}
    idx_to_word = {i + 1: w for i, w in enumerate(all_words)}

    return Vocab(word_to_idx, idx_to_word)


def token_featurizer(nodes, tokenizer):
    out = []
    for node in nodes:
        text = " ".join(node.xpath(".//text()")).lower()
        tokens = tokenizer.tokenize(text)
        out.extend(tokens)

    return out


def char_tokenizer(nodes, tokenizer):
    out = []
    for node in nodes:
        text = " ".join(node.xpath(".//text()")).lower()
        tokens = [char for token in tokenizer.tokenize(text) for char in token]
        out.extend(tokens)

    return out


def to_onehot(idx: int, n: int) -> List[int]:
    """Convert an integer into an n-dimensional one-hot vector.

    For example, to_onehot(2, 5) -> [0, 0, 1, 0, 0].
    :param idx: The non-zero index.
    :param n: The dimensionality of the output vector.
    :returns: A list of `n` elements.
    """
    out = [0] * n
    out[idx] = 1
    return out
