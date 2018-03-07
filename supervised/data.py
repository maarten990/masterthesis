import os.path
import pickle
import random
import re
from collections import defaultdict
from glob import glob
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import nltk
import numpy as np
from lxml import etree
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

random.seed(100)


class Vocab:
    def __init__(self, token_to_idx: Dict[str, int], idx_to_token: Dict[int, str]) -> None:
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token


class GermanDataset(Dataset):
    def __init__(self, folder: str, filenames: List[str], num_clusterlabels: int,
                 window_size: int, window_label_idx: int = 0) -> None:
        self.paths = [os.path.join(folder, fname) for fname in filenames]
        self.vocab = create_dictionary(self.paths)
        self.num_clusterlabels = num_clusterlabels
        self.window_size = window_size
        self.window_label_idx = window_label_idx

        lengths = []
        # subtract the elements that get dropped off due to the window size
        window_loss = window_size - 1
        for path in self.paths:
            tree = load_xml_from_disk(path)
            lengths.append(len(tree.xpath('/pdf2xml/page/text')) - window_loss)

        self.boundaries = np.cumsum(lengths)
            
    def __len__(self) -> int:
        return self.boundaries[-1]

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        file_idx = len([b for b in self.boundaries if idx >= b])
        if file_idx == 0:
            file_offset = idx
        else:
            file_offset = idx - self.boundaries[file_idx - 1]
        tree = load_xml_from_disk(self.paths[file_idx])

        end = file_offset + self.window_size
        elements = tree.xpath('/pdf2xml/page/text')[file_offset:end]
        return self.vectorize_window(elements)

    def vectorize_window(self, window: List[etree._Element]) -> Dict[str, np.ndarray]:
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

        return {'data': np.array(X),
                'speaker_data': np.array(X_speaker),
                'cluster_data': np.array(clusterlabels),
                'label': np.array([y])}


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
    all_words = set() # type: Set[str]

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


def collate(samples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    out = defaultdict(list) # type: Dict[str, np.ndarray]
    for sample in samples:
        for key, item in sample.items():
            out[key].append(item)
        
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
