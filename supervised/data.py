import os.path
import pickle
import re
from functools import lru_cache
from glob import glob
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import nltk
import numpy as np
import torch
from torch.autograd import Variable
from lxml import etree
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

np.random.seed(100)


class Vocab:
    def __init__(self, token_to_idx: Dict[str, int], idx_to_token: Dict[int, str]) -> None:
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token


Sample = Dict[int, Dict[str, np.ndarray]]
class GermanDataset(Dataset):
    def __init__(self, files: List[str], num_clusterlabels: int,
                 window_size: int, window_label_idx: int = 0) -> None:
        self.files = files
        self.vocab = create_dictionary(self.files)
        self.num_clusterlabels = num_clusterlabels
        self.window_size = window_size
        self.window_label_idx = window_label_idx

        lengths: List[int] = []
        # subtract the elements that get dropped off due to the window size
        window_loss = window_size - 1
        for path in self.files:
            tree = load_xml_from_disk(path)
            lengths.append(len(tree.xpath('/pdf2xml/page/text')) - window_loss)

        self.boundaries = np.cumsum(lengths)
            
    def __len__(self) -> int:
        return self.boundaries[-1]

    def __getitem__(self, idx: int) -> Sample:
        file_idx = len([b for b in self.boundaries if idx >= b])
        if file_idx == 0:
            file_offset = idx
        else:
            file_offset = idx - self.boundaries[file_idx - 1]
        tree = load_xml_from_disk(self.files[file_idx])

        end = file_offset + self.window_size + 1
        elements = tree.xpath('/pdf2xml/page/text')[file_offset:end]
        sample = self.vectorize_window(elements)

        assert len(sample) == 1
        return sample

    def vectorize_window(self, window: List[etree._Element]) -> Sample:
        tokenizer = nltk.tokenize.WordPunctTokenizer()
        tokens = token_featurizer(window, tokenizer)
        y = get_label(window[self.window_label_idx])

        # lowercase as the tokens will also be lowercased
        speaker_tokens = tokenizer.tokenize(
            get_speaker(window[self.window_label_idx]).lower())

        X = [self.vocab.token_to_idx.get(token, 0) for token in tokens]
        
        X_speaker = [1 if token in speaker_tokens else 0 for token in tokens]

        clusterlabels = to_onehot(
            int(window[self.window_label_idx].attrib['clusterLabel']),
            self.num_clusterlabels)

        return {len(X): {'data': np.array(X),
                         'speaker_data': np.array(X_speaker),
                         'cluster_data': np.array(clusterlabels),
                         'label': np.array([y])}}

class GermanDatasetInMemory(GermanDataset):
    def __init__(self, files: List[str], num_clusterlabels: int,
                 num_positive: int, num_negative: int, window_size: int,
                 window_label_idx: int = 0) -> None:
        super().__init__(files, num_clusterlabels, window_size, window_label_idx)
        self.samples: List[Sample] = []

        pos = 0
        neg = 0
        for i in tqdm(range(super().__len__()), desc='Loading samples'):
            self.samples.append(super().__getitem__(i))
            if (list(self.samples[-1].values())[0]['label'] == 1).all():
                pos += 1
            else:
                neg += 1

            if pos >= num_positive and neg >= num_negative:
                break

        if len(self.samples) < num_positive + num_negative:
            print(f'Warning: could only obtain {len(self.samples)} samples')
        else:
            self.subsample(num_positive, num_negative)

    def subsample(self, num_positive: int, num_negative: int) -> None:
        # first, divide the samples in positive and negative samples
        positives: List[int] = []
        negatives: List[int] = []
        for i, sample in enumerate(self.samples):
            # the key (length) is not relevant, and we know there's only item
            for _, data in sample.items():
                if (data['label'] == 1).all():
                    positives.append(i)
                else:
                    negatives.append(i)

        # then subsample the negative samples until the amount is equal
        neg_diff = len(negatives) - num_negative
        pos_diff = len(positives) - num_positive
        neg_discard = np.random.choice(negatives, neg_diff, replace=False)
        pos_discard = np.random.choice(positives, pos_diff, replace=False)
        self.samples = [sample for i, sample in enumerate(self.samples)
                        if i not in neg_discard and i not in pos_discard]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def get_iterator(dataset: Dataset, buckets: List[int] = [40], batch_size: int = 32 ) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, collate_fn=CollateWithBuckets(buckets))


class CollateWithBuckets:
    def __init__(self, buckets: List[int]) -> None:
        self.buckets = buckets

    def __call__(self, samples: List[Sample]) -> Sample:
        return self.bucket(samples)

    def bucket(self, samples: List[Sample]) -> Sample:
        out: Sample = {}
        for sample in samples:
            for size, s in sample.items():
                for bucket_size in self.buckets:
                    if size <= bucket_size:
                        if bucket_size in out:
                            out[bucket_size] = self.concat_samples(
                                out[bucket_size], self.pad(s, bucket_size - size))
                        else:
                            out[bucket_size] = self.pad(s, bucket_size - size)

                        break
                else:
                    # truncate
                    bucket_size = self.buckets[-1]
                    if bucket_size in out:
                        out[bucket_size] = self.concat_samples(
                            out[bucket_size], self.pad(s, bucket_size - size))
                    else:
                        out[bucket_size] = self.pad(s, bucket_size - size)


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


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    return np.expand_dims(arr, 0) if len(arr.shape) < 2 else arr


def to_tensors(sample: Sample) -> Sample:
    out: Sample = {}

    for size, sample_dict in sample.items():
        out[size] = {'data': Variable(torch.from_numpy(sample_dict['data'])).long(),
                     'speaker_data': Variable(torch.from_numpy(sample_dict['speaker_data'])).long(),
                     'cluster_data': Variable(torch.from_numpy(sample_dict['cluster_data'])).float(),
                     'label': Variable(torch.from_numpy(sample_dict['label'])).float()}

    return out


@lru_cache(maxsize=8192)
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

    For example, make_categorical(2, 5) -> [0, 0, 1, 0, 0].
    :param idx: The non-zero index.
    :param n: The dimensionality of the output vector.
    :returns: A list of `n` elements.
    """
    out = [0] * n
    out[idx] = 1
    return out


'''
def pad_sequences_to_buckets(data: Dict[str, np.ndarray], bucket_sizes: List[int]) -> Dict[str, np.ndarray]:
    """
    Pad the list of variable-length sequences X to arrays with widths
    corresponding to the specified buckets.
    If the last bucket is -1, it will be set to the largest occurring sequence
    length.

    Returns a list of n arrays, n being the number of buckets.
    """
    if bucket_sizes[-1] == -1:
        bucket_sizes[-1] = max(len(seq) for seq in X)

    X = data['data']
    y = data['labels']
    cluster_labels = data['cluster_data']

    buckets = [[] for _ in bucket_sizes] # type: List[List[List[float]]]
    labels = [[] for _ in bucket_sizes] # type: List[List[Union[List[int], int]]]
    clusters = [[] for _ in bucket_sizes] # type: List[List[List[int]]]

    for seq, label, cluster in zip(X, y, cluster_labels):
        for bucket_size, bucket, label_bucket, cluster_bucket in zip(bucket_sizes, buckets, labels, clusters):
            if len(seq) <= bucket_size:
                diff = bucket_size - len(seq)
                bucket.append(np.pad(seq, (0, diff), 'constant'))
                cluster_bucket.append(cluster)

                if type(label) == list:
                    label_bucket.append(np.pad(label, (0, diff), 'constant'))
                else:
                    label_bucket.append(label)

                break
        else:
            # If the for-loop didn't break, the sequence will need to be
            # truncated into the largest bucket.
            buckets[-1].append(seq[:bucket_sizes[-1]])
            clusters[-1].append(cluster)
            if type(label) == list:
                labels[-1].append(label[:bucket_sizes[-1]])
            else:
                labels[-1].append(label)

    return ([np.array(bucket) for bucket in buckets],
            [np.array(label_bucket) for label_bucket in labels],
            [np.array(cluster_bucket) for cluster_bucket in clusters])
'''
