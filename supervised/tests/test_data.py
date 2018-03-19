import random
import numpy as np
from .. import data

import pytest


def test_create_dictionary():
    vocab = data.create_dictionary(['../clusterlabeled-5/18001.xml'])

    # both dictionaries should of course be the same length as they map to
    # one-another
    assert len(vocab.idx_to_token) == len(vocab.token_to_idx)

    # the index 0 is reserved for unknown tokens
    assert 0 not in vocab.idx_to_token

    # make sure that each each index translates back and forth correctly
    for i in range(1, len(vocab.idx_to_token) + 1):
        word = vocab.idx_to_token[i]
        reverse_idx = vocab.token_to_idx[word]
        assert i == reverse_idx


@pytest.fixture(scope='module')
def dataset():
    num_clusters = 5
    num_pos = 10
    num_neg = 100
    d = data.GermanDatasetInMemory([f'../clusterlabeled-{num_clusters}/18001.xml',
                                    f'../clusterlabeled-{num_clusters}/18001.xml'],
                                   num_clusters, num_pos, num_neg, 3)
    assert len(d) == num_pos + num_neg
    return d


def batch_test(batch, vocab):
    for size, s in batch.items():
        data = s['data']
        assert data.shape[-1] == size

        if len(data.shape) == 1:
            data = np.expand_dims(data, 0)

        for i in range(data.shape[0]):
            tokens = [vocab.idx_to_token.get(idx, '<unknown>') for idx in data[i, :]]
            assert s['label'][i] == 0 or ':' in tokens


def test_dataset(dataset):
    for sample in dataset:
        batch_test(sample, dataset.vocab)


def test_dataloader(dataset):
    buckets = [10, 20, 40]
    batch_size = 32
    dl = data.get_iterator(dataset, buckets=buckets, batch_size=batch_size)
    not_batch_size = 0
    for batch in dl:
        batch_test(batch, dataset.vocab)
        size_of_batch = 0
        for size, s in batch.items():
            assert size in buckets
            size_of_batch += s['data'].shape[0]

        if size_of_batch != batch_size:
            not_batch_size += 1

    assert not_batch_size <= 1


def test_split(dataset):
    assert False
