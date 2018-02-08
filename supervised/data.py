import os.path
import pickle
import random
import re
from glob import glob

import nltk
import numpy as np
from lxml import etree
from tqdm import tqdm

XMLNS = {'pm': 'http://www.politicalmashup.nl',
         'dc': 'http://purl.org/dc/elements/1.1'}


class Vocab:
    def __init__(self, token_to_idx, idx_to_token):
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token


class Data:
    def __init__(self, X=None, y=None, speakers=None, vocab=None,
                 clusterLabels=None):
        self.X = X
        self.y = y
        self.speakers = speakers
        self.vocab = vocab
        self.clusterLabels = clusterLabels


def pickler(func):
    def wrapped(*args, **kwargs):
        args_str = '_'.join(str(arg) for arg in args)
        kwargs_str = '_'.join(f'{key}_{val}' for key, val in kwargs)

        # remove any illegal/stupid characters
        args_str = re.sub(r'[/\\\*]', '', args_str)
        kwargs_str = re.sub(r'[/\\\*]', '', kwargs_str)

        pkl_path = os.path.join('pickle', f'{func.__name__}_{args_str}_{kwargs_str}.pkl')

        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        else:
            out = func(*args, **kwargs)
            with open(pkl_path, 'wb') as f:
                pickle.dump(out, f)

            return out

    return wrapped


def load_from_disk(folder, pattern):
    """ Load each xml file for the given folder as an etree. """
    parser = etree.XMLParser(ns_clean=True, encoding='utf-8')

    for file in glob(os.path.join(folder, pattern)):
        with open(file, 'r', encoding='utf-8') as f:
            xml = etree.fromstring(f.read().encode('utf-8'), parser)
            yield xml


def get_label(node):
    is_speech = node.attrib['is-speech']
    return 1 if is_speech == 'true' else 0


def get_speaker(node):
    is_speech = node.attrib['is-speech']

    if is_speech == 'true':
        return node.attrib['speaker']
    else:
        return ''


@pickler
def create_dictionary(folder, pattern):
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    all_words = set()

    for xml in tqdm(load_from_disk(folder, pattern), desc='Creating dictionary'):
        text = ' '.join(xml.xpath('//text//text()')).lower()
        tokens = tokenizer.tokenize(text)
        all_words |= set(tokens)

    word_to_idx = {w: i+1 for i, w in enumerate(all_words)}
    idx_to_word = {i+1: w for i, w in enumerate(all_words)}

    return Vocab(word_to_idx, idx_to_word)


def sliding_window(folder, pattern, n, prune_ratio, label_pos=0, vocab=None,
                   withClusterLabels=False):
    """
    Return a sliding window representation over the documents with the
    given feature transformation.
    """
    if not vocab:
        vocab = create_dictionary(folder, pattern)

    tokenizer = nltk.tokenize.WordPunctTokenizer()

    inputs = []
    is_speech = []
    speakers = []
    clusterLabels = []

    for xml in load_from_disk(folder, pattern):
        nodes = xml.xpath('//text')
        for window in zip(*(nodes[i:] for i in range(n))):
            tokens = token_featurizer(window, tokenizer)
            y = get_label(window[label_pos])

            # lowercase as the tokens will also be lowercased
            speaker_tokens = tokenizer.tokenize(get_speaker(window[label_pos]).lower())

            # skip empty lines
            if len(tokens) == 0:
                continue

            # random pruning for negative labels
            if y == 0 and random.random() > prune_ratio:
                continue

            inputs.append([vocab.token_to_idx[token] if token in vocab.token_to_idx else 0
                           for token in tokens])
            is_speech.append(y)
            speakers.append([1 if token in speaker_tokens else 0 for token in tokens])

            if withClusterLabels:
                clusterLabels.append([int(node.attrib['clusterLabel'])
                                      for node in window])

    return Data(X=inputs, y=np.array(is_speech), speakers=speakers, vocab=vocab,
                clusterLabels=clusterLabels)


def token_featurizer(nodes, tokenizer):
    out = []
    for node in nodes:
        text = ' '.join(node.xpath('.//text()')).lower()
        tokens = tokenizer.tokenize(text)
        out.extend(tokens)

    return out


def pad_sequences(X, y, bucket_sizes):
    """
    Pad the list of variable-length sequences X to arrays with widths
    corresponding to the specified buckets.
    If the last bucket is -1, it will be set to the largest occurring sequence
    length.

    Returns a list of n arrays, n being the number of buckets.
    """
    if bucket_sizes[-1] == -1:
        bucket_sizes[-1] = max(len(seq) for seq in X)

    buckets = [[] for _ in bucket_sizes]
    labels = [[] for _ in bucket_sizes]
    for seq, label in zip(X, y):
        for bucket_size, bucket, label_bucket in zip(bucket_sizes, buckets, labels):
            if len(seq) <= bucket_size:
                diff = bucket_size - len(seq)
                bucket.append(np.pad(seq, (0, diff), 'constant'))

                if type(label) == list:
                    label_bucket.append(np.pad(label, (0, diff), 'constant'))
                else:
                    label_bucket.append(label)

                break
        else:
            # If the for-loop didn't break, the sequence will need to be
            # truncated into the largest bucket.
            buckets[-1].append(seq[:bucket_sizes[-1]])
            if type(label) == list:
                labels[-1].append(label[:bucket_sizes[-1]])
            else:
                labels[-1].append(label)

    return ([np.array(bucket) for bucket in buckets],
            [np.array(label_bucket) for label_bucket in labels])
