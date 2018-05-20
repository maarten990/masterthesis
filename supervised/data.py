"""Contains functions for loading and manipulating training data."""


from copy import copy
from enum import auto, Enum
from typing import Dict, Iterator, List, Optional, Tuple, Union

from lxml import etree
import nltk
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm

np.random.seed(100)


class ClusterFmt(Enum):
    # only uses the cluster label of the central element of the window
    ONLY_CENTRAL = auto()

    # a concatenated sequence of the categorical labels of the whole window
    FULL_WINDOW = auto()

    # Same as full window, but with the indices instead of a categorical vector.
    # Meant for use as input to a Pytorch embedding layer.
    FULL_WINDOW_ONLY_IDX = auto()


def only_central_fn(window, central_idx, num_clusterlabels):
    return to_onehot(
        int(window[central_idx].attrib['clusterLabel']) if 'clusterLabel' in window[central_idx].attrib else 0,
        num_clusterlabels
    )


def full_window_fn(window, central_idx, num_clusterlabels):
    l = [to_onehot(int(w.attrib['clusterLabel']) if 'clusterLabel' in w.attrib else 0,
                   num_clusterlabels)
         for w in window]
    return np.concatenate(l)


def full_window_only_idx_fn(window, central_idx, num_clusterlabels):
    return [int(w.attrib['clusterLabel']) if 'clusterLabel' in w.attrib else 0
            for w in window]


class Vocab:
    def __init__(self, token_to_idx: Dict[str, int], idx_to_token: Dict[int, str]) -> None:
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token


Sample = Dict[int, Dict[str, np.ndarray]]


class GermanDataset(Dataset):
    def __init__(self, files: List[str], num_clusterlabels: int,
                 num_positive: int, num_negative: int, window_size: int,
                 window_label_idx: int = 0, vocab: Optional[Vocab]=None,
                 bag_of_words: bool = False,
                 cluster_fmt: ClusterFmt = ClusterFmt.FULL_WINDOW_ONLY_IDX) -> None:
        self.files = files
        self.vocab = create_dictionary(self.files) if not vocab else vocab
        self.num_clusterlabels = num_clusterlabels
        self.window_size = window_size
        self.window_label_idx = window_label_idx
        self.samples: List[Sample] = []
        self.bag_of_words = bag_of_words
        self.cluster_fmt = {
            ClusterFmt.FULL_WINDOW: full_window_fn,
            ClusterFmt.FULL_WINDOW_ONLY_IDX: full_window_only_idx_fn,
            ClusterFmt.ONLY_CENTRAL: only_central_fn
        }[cluster_fmt]
        n_pos = 0
        n_neg = 0

        with tqdm(total=num_positive) as pbar:
            for file in files:
                xml = load_xml_from_disk(file)
                pos = xml.xpath('/pdf2xml/page/text[@is-speech="true"]')
                neg = xml.xpath('/pdf2xml/page/text[@is-speech="false"]')

                for p in pos:
                    window = xml_window(p, window_label_idx, window_size)
                    if window:
                        self.samples.append(self.vectorize_window(window))
                        n_pos += 1
                        pbar.update(1)
                for n in neg:
                    window = xml_window(n, window_label_idx, window_size)
                    if window:
                        self.samples.append(self.vectorize_window(window))
                        n_neg += 1

                if n_pos >= num_positive and n_neg >= num_negative:
                    break

        self.subsample(num_positive, num_negative)

    def get_pos_neg(self) -> Tuple[List[int], List[int]]:
        positives: List[int] = []
        negatives: List[int] = []
        for i, sample in enumerate(self.samples):
            # the key (length) is not relevant, and we know there's only one item
            for _, data in sample.items():
                if (data['label'] == 1).all():
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
            # the key (length) is not relevant, and we know there's only one item
            for _, data in sample.items():
                labels.append(data['label'])

        return labels

    def subsample(self, num_positive: int, num_negative: int) -> None:
        positives, negatives = self.get_pos_neg()
        neg_diff = len(negatives) - num_negative
        pos_diff = len(positives) - num_positive
        neg_discard = np.random.choice(negatives, neg_diff, replace=False)
        pos_discard = np.random.choice(positives, pos_diff, replace=False)
        self.samples = [sample for i, sample in enumerate(self.samples)
                        if i not in neg_discard and i not in pos_discard]

        print(f'Retrieved {len(positives)} positive samples, {len(negatives)} negative samples.')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def split(self, train: Tuple[int, int], test: Optional[Tuple[int, int]] = None) -> Tuple[Dataset, Dataset]:
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
        pos_indices = np.random.choice(positives, train[0] + test[0], replace=False)
        neg_indices = np.random.choice(negatives, train[1] + test[1], replace=False)

        train_indices = np.concatenate((pos_indices[:train[0]], neg_indices[:train[0]]))
        test_indices = np.concatenate((pos_indices[:test[0]], neg_indices[:test[0]]))
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        if return_test:
            return DataSubset(self, train_indices), DataSubset(self, test_indices)
        else:
            return DataSubset(self, train_indices)

    def kfold(self, k: int = 10) -> Iterator[Dataset]:
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

    def vectorize_window(self, window: List[etree._Element]) -> Sample:
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+|[^\w\s]')
        tokens = token_featurizer(window, tokenizer)
        y = get_label(window[self.window_label_idx])

        # lowercase as the tokens will also be lowercased
        speaker_tokens = tokenizer.tokenize(
            get_speaker(window[self.window_label_idx]).lower())

        if self.bag_of_words:
            X = ' '.join(tokens)
        else:
            X = np.array([self.vocab.token_to_idx.get(token, 0) for token in tokens])

        X_speaker = [1 if token in speaker_tokens else 0 for token in tokens]

        clusterlabels = self.cluster_fmt(window, self.window_label_idx, self.num_clusterlabels)

        return {len(X): {'data': X,
                         'speaker_data': np.array(X_speaker),
                         'cluster_data': np.array(clusterlabels),
                         'label': np.array([y])}}


class DataSubset(GermanDataset):
    def __init__(self, data: GermanDataset, indices: List[int]) -> None:
        self.samples = [data.samples[i] for i in indices]
        self.vocab = data.vocab
        self.num_clusterlabels = data.num_clusterlabels


def xml_window(node: etree._Element, n_before: int, size: int) -> List[etree._Element]:
    start = node
    for _ in range(n_before):
        prev = start.getprevious()
        if prev is not None:
            start = start = prev
        else:
            break

    out = []
    for _ in range(size):
        if start is None:
            return []
        out.append(start)
        start = start.getnext()

    return out


def get_iterator(dataset: Dataset, buckets: List[int] = [40], batch_size: int = 32) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, collate_fn=CollateWithBuckets(buckets))


class CollateWithBuckets:
    def __init__(self, buckets: List[int]) -> None:
        self.buckets = buckets

    def __call__(self, samples: List[Sample]) -> Sample:
        return self.bucket(samples)

    def bucket(self, samples: List[Sample]) -> Sample:
        buckets = copy(self.buckets)
        if buckets[-1] == -1:
            buckets[-1] = max([size for sample in samples for size in sample.keys()])

        out: Sample = {}
        for sample in samples:
            for size, s in sample.items():
                for bucket_size in buckets:
                    if size <= bucket_size:
                        if bucket_size in out:
                            out[bucket_size] = self.concat_samples(
                                out[bucket_size], self.pad(s, bucket_size - size))
                        else:
                            out[bucket_size] = ensure_2d(self.pad(s, bucket_size - size))

                        break
                else:
                    # truncate
                    bucket_size = buckets[-1]
                    if bucket_size in out:
                        out[bucket_size] = self.concat_samples(
                            out[bucket_size], self.pad(s, bucket_size - size))
                    else:
                        out[bucket_size] = ensure_2d(self.pad(s, bucket_size - size))

        return out

    def pad(self, sample_dict: Dict[str, np.ndarray], amount: int) -> Dict[str, np.ndarray]:
        if amount >= 0:
            return {'data': np.pad(sample_dict['data'], (0, amount), 'constant'),
                    'speaker_data': np.pad(sample_dict['speaker_data'], (0, amount), 'constant'),
                    'cluster_data': sample_dict['cluster_data'],
                    'label': sample_dict['label']}
        else:
            return {'data': sample_dict['data'][:amount],
                    'speaker_data': sample_dict['speaker_data'][:amount],
                    'cluster_data': sample_dict['cluster_data'],
                    'label': sample_dict['label']}

    def concat_samples(self, sample1: Dict[str, np.ndarray], sample2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {key: np.append(ensure_2d(sample1[key]), ensure_2d(sample2[key]), 0)
                for key in sample1.keys()}


def ensure_2d(arr: Union[np.ndarray, Dict[str, np.ndarray]]) -> np.ndarray:
    if isinstance(arr, dict):
        return {key: ensure_2d(value) for key, value in arr.items()}
    else:
        return np.expand_dims(arr, 0) if len(arr.shape) < 2 else arr


def to_tensors(sample: Sample) -> Sample:
    out: Sample = {}

    for size, sample_dict in sample.items():
        out[size] = {'data': Variable(torch.from_numpy(sample_dict['data'])).long(),
                     'speaker_data': Variable(torch.from_numpy(sample_dict['speaker_data'])).long(),
                     'cluster_data': Variable(torch.from_numpy(sample_dict['cluster_data'])).long(),
                     'label': Variable(torch.from_numpy(sample_dict['label'])).float()}

    return out


def to_gpu(sample: Sample) -> Sample:
    if torch.cuda.is_available():
        out: Sample = {}
        for size, sample_dict in sample.items():
            out[size] = {'data': sample_dict['data'].cuda(),
                         'speaker_data': sample_dict['speaker_data'].cuda(),
                         'cluster_data': sample_dict['cluster_data'].cuda(),
                         'label': sample_dict['label'].cuda()}

        return out
    else:
        return sample


def to_cpu(sample: Sample) -> Sample:
    out: Sample = {}
    for size, sample_dict in sample.items():
        out[size] = {'data': sample_dict['data'].cpu(),
                     'speaker_data': sample_dict['speaker_data'].cpu(),
                     'cluster_data': sample_dict['cluster_data'].cpu(),
                     'label': sample_dict['label'].cpu()}

    return out


def load_xml_from_disk(path: str) -> etree._Element:
    parser = etree.XMLParser(ns_clean=True, encoding='utf-8')
    with open(path, 'r', encoding='utf-8') as f:
        return etree.fromstring(f.read().encode('utf-8'), parser)


def get_label(node: etree._Element) -> int:
    is_speech = node.attrib['is-speech']
    return 1 if is_speech == 'true' else 0


def get_speaker(node: etree._Element) -> str:
    is_speech = node.attrib['is-speech']

    if is_speech == 'true':
        return node.attrib['speaker']
    else:
        return ''


def create_dictionary(paths: List[str]) -> Vocab:
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    all_words: Set[str] = set()

    for path in tqdm(paths, desc='Creating dictionary'):
        xml = load_xml_from_disk(path)
        text = ' '.join(xml.xpath('/pdf2xml/page/text/text()')).lower()
        tokens = tokenizer.tokenize(text)
        all_words |= set(tokens)

    word_to_idx = {w: i+1 for i, w in enumerate(all_words)}
    idx_to_word = {i+1: w for i, w in enumerate(all_words)}

    return Vocab(word_to_idx, idx_to_word)


def token_featurizer(nodes, tokenizer):
    out = []
    for node in nodes:
        text = ' '.join(node.xpath('.//text()')).lower()
        tokens = tokenizer.tokenize(text)
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
