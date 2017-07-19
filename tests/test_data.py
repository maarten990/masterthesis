import random
import numpy as np
from .. import data


def test_create_dictionary():
    vocab = data.create_dictionary('training_data', '1800*.xml')

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


def test_sliding_window():
    X_train, y_is_speech, Y_speaker, vocab = data.sliding_window('tests', 'test.xml', 2, 1)

    # text.xml contains 4 speeches; with a window size of 2 that means there are 3 windows
    assert len(X_train) == 3

    # test the expected windows
    expected = ['text 1 text 2', 'text 2 speaker : text 3', 'speaker : text 3 text 4']
    for X, e in zip(X_train, expected):
        assert ' '.join(vocab.idx_to_token[x] for x in X) == e

    # the 3rd node contains a speech by 'Mister Speaker'
    assert np.all(y_is_speech == np.array([0, 0, 1]))

    # test the expected speakers, which are boolean arrays of the same size as
    # the entries in X_train, indicating wether each tokens is part of the name
    for i, (speaker, tokens) in enumerate(zip(Y_speaker, X_train)):
        assert len(speaker) == len(tokens)

        speaker = np.array(speaker)
        if i == 2:
            assert speaker[0] == 1 and np.all(speaker[1:] == 0)
        else:
            assert np.all(speaker == 0)


def test_pad_sequences():
    # Length: count
    # 1: 2
    # 3: 5
    # 5: 3
    X = [[0] for _ in range(2)] + [[0, 0, 0] for _ in range(5)] \
        + [[0, 0, 0, 0, 0] for _ in range(3)]

    random.shuffle(X)

    # labels are the lengths of the sequences
    y = [len(seq) for seq in X]

    sizes = [3, -1]
    buckets, labels = data.pad_sequences(X, y, sizes)
    assert len(buckets) == len(sizes)
    assert len(buckets) == len(labels)
    assert buckets[0].shape == (7, sizes[0])
    assert buckets[1].shape == (3, 5)
    for bucket, label in zip(buckets, labels):
        assert np.all(label <= (bucket.shape[1]))

    sizes = [2, 4]
    buckets, labels = data.pad_sequences(X, y, sizes)
    assert len(buckets) == len(sizes)
    assert len(buckets) == len(labels)
    assert buckets[0].shape == (2, sizes[0])
    assert buckets[1].shape == (8, sizes[1])
    for bucket, label in zip(buckets, labels):
        assert np.all(label <= (bucket.shape[1]))