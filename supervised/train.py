"""
Train the neural networks.

Usage:
train.py (rnn | cnn | speaker) <folder> <trainpattern> <testpattern>
    [--epochs=<n>] [--dropout=<ratio>] [--with_labels] [--eval=<file>]
train.py (-h | --help)

Options:
    -h --help                Show this screen
    --epochs=<n>             Number of epochs to train for [default: 100]
    --dropout=<ratio>        The dropout ratio between 0 and 1 [default: 0.5]
    --with_labels            Include computed cluster labels.
    -e <file> --eval=<file>  Evaluate after every epoch and write the output to <file>.
"""


import os.path
import re
from collections import namedtuple
from docopt import docopt

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from tabulate import tabulate
from torch.autograd import Variable
from tqdm import trange

from data import sliding_window, pad_sequences
from models import LSTMClassifier, CNNClassifier, NameClassifier, cluster_factory

Datatuple = namedtuple('Datatuple', ['X_is_speech', 'X_speaker', 'y_is_speech', 'Y_speaker'])


def get_filename(network, pattern):
    """
    Get the filename to use for pickling a classifier.

    network: the type of network (i.e. cnn, rnn or speaker)
    pattern: the glob pattern used for gathering training data

    Returns: a string representing a file path
    """
    path = f'pickle/{network}_{pattern}'
    path = re.sub(r'[*.]', '', path) + '.pkl'
    return path


def load_model(filename):
    """
    Load a model from the given path.
    """
    if os.path.exists(filename):
        modelfn, optimfn, model_state, optim_state = torch.load(filename)
        model = modelfn()
        optim = optimfn(model.parameters())

        model.load_state_dict(model_state)
        optim.load_state_dict(optim_state)
        return model, optim
    else:
        return None


def write_losses(losses, basename):
    batch, epoch = losses

    with open('evals_' + basename, 'w') as f:
        f.write('\n'.join([str(l) for l in batch]))

    with open('losses_' + basename, 'w') as f:
        f.write('\n'.join([str(l) for l in epoch]))


def get_clf_data(folder, trainpattern, testpattern, buckets):
    """
    Return data meant for speech classification

    folder: the folder containing the xml files
    trainpattern: a glob pattern for selecting training files, e.g. '1800*.xml'
    testpattern: a glob pattern for selecting training files, e.g. '1800*.xml'
    buckets: sequence length buckets to pad the data to

    Returns:
    (training samples, testing samples, training labels, testing labels,
     the vocabulary that was generated)

    The samples and labels are lists of numpy arrays, each array representing
    a bucket.
    """
    data = sliding_window(folder, trainpattern, 2, 0.1, withClusterLabels=True)
    X, y = data.X, data.y
    vocab = data.vocab
    cluster_labels = data.clusterLabels
    Xb, yb, cb = pad_sequences(X, y, buckets, cluster_labels=cluster_labels)

    data = sliding_window(folder, testpattern, 2, 0.1, vocab=vocab, withClusterLabels=True)
    Xt, yt = data.X, data.y
    cluster_labels = data.clusterLabels
    Xtb, ytb, ctb = pad_sequences(Xt, yt, buckets, cluster_labels=cluster_labels)

    for i, X in enumerate(Xb):
        train_samples = X.shape[0] if len(X) > 0 else 0
        test_samples = sum(bucket.shape[0] if len(bucket) > 0 else 0 for bucket in Xtb)
        seqlen = Xtb[i].shape[1] if len(Xtb[i]) > 0 else 0
        print('Speech classifier bucket {}: {} training samples, {} testing samples, sequence length {}'.format(
            i, train_samples, test_samples, seqlen))

    return Xb, Xtb, yb, ytb, cb, ctb, vocab


def get_speaker_data(folder, trainpattern, testpattern, seqlen):
    """
    Return data meant for speaker extraction.

    folder: the folder containing the xml files
    trainpattern: a glob pattern for selecting training files, e.g. '1800*.xml'
    testpattern: a glob pattern for selecting training files, e.g. '1800*.xml'
    seqlen: the sequence length to which to pad/truncate the samples

    Returns:
    (training samples, testing samples, training labels, testing labels,
     the vocabulary that was generated)

    The samples and labels are wrapped in a single-element list for
    compatibility with the functions expecting bucketed data.
    """
    # X, y, Y, vocab = sliding_window(folder, trainpattern, 2, 0.1)
    data = sliding_window(folder, trainpattern, 2, 0.1)
    X, y, Y, vocab = data.X, data.y, data.speakers, data.vocab
    X, Y = pad_sequences(X, Y, [seqlen])

    data = sliding_window(folder, testpattern, 2, 0.1, vocab=vocab)
    Xt, yt, Yt = data.X, data.y, data.speakers
    Xt, Yt = pad_sequences(Xt, Yt, [seqlen])

    # remove the bucketing since we don't use it for this network
    X, Y, Xt, Yt = X[0], Y[0], Xt[0], Yt[0]

    # filter the spkr extraction data to only positive samples
    X = X[y == 1, :]
    Y = Y[y == 1, :]
    Xt = Xt[yt == 1, :]
    Yt = Yt[yt == 1, :]

    print('Speaker extraction: {} training samples, {} testing samples, sequence length {}'.format(
        X.shape[0], Xt.shape[0], X.shape[1]))

    return [X], [Xt], [Y], [Yt], vocab


def train(model, optimizer, X_buckets, y_buckets, cluster_buckets, epochs=100, batch_size=32, eval_fn=None):
    model.train()

    epoch_losses = []
    epoch_evals = []
    t = trange(epochs, desc='Training')
    for _ in t:
        epoch_loss = torch.zeros(1).float()

        for X_train, y_train, c_train in zip(X_buckets, y_buckets, cluster_buckets):
            for i in range(0, X_train.shape[0], batch_size):
                X = Variable(torch.from_numpy(X_train[i:i+32, :])).long()
                y = Variable(torch.from_numpy(y_train[i:i+32])).float()
                c = Variable(torch.from_numpy(c_train[i:i+32, :])).float()

                y_pred = model(X, c)
                loss = model.loss(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.data.cpu()

        loss = epoch_loss[0]
        epoch_losses.append(loss)
        loss_delta = epoch_losses[-1] - epoch_losses[-2] if len(epoch_losses) > 1 else 0
        t.set_postfix({'loss': loss,
                       'Δloss': loss_delta})

        if eval_fn:
            epoch_evals.append(eval_fn(model))
            model.train()

    model.eval()
    return (epoch_evals, epoch_losses), optimizer


def evaluate_clf(model, Xb, cb, yb, batch_size=32, silent=False):
    """
    Evaluate the trained model.
    Xb, cb, yb: bucketed lists of training, cluster and test data
    """
    model = model.eval()
    predictions = []
    true = []

    for X, c, y in zip(Xb, cb, yb):
        for i in range(0, X.shape[0], batch_size):
            Xvar = Variable(torch.from_numpy(X[i:i+32, :])).long()
            ybatch = y[i:i+32]
            cvar = Variable(torch.from_numpy(c[i:i+32, :])).float()

            pred = model(Xvar, cvar)
            pred = pred.cpu().squeeze().data.numpy()
            pred = np.where(pred > 0.5, 1, 0)
            predictions.extend(pred)
            true.extend(ybatch)

    table = []
    table.append(['f1', f1_score(true, predictions)])
    table.append(['Speech recall', recall_score(true, predictions)])
    table.append(['Speech precision', precision_score(true, predictions)])

    if not silent:
        print()
        print(tabulate(table))

    return f1_score(true, predictions)


def evaluate_spkr(model, Xb, yb, idx_to_token):
    model.eval()

    correct = 0
    for X, y in zip(Xb, yb):
        Xvar = Variable(torch.from_numpy(X)).long()
        predictions = model(Xvar)

        for i in range(X.shape[0]):
            full_string = X[i, :]
            true = y[i, :]
            pred = predictions.cpu().data.numpy()[i, :]

            true_words = full_string[true > 0.5]
            pred_words = full_string[pred > 0.5]

            if np.all(true_words == pred_words):
                correct += 1

    print()
    print(f'Speaker accuracy: {correct / X.shape[0]}')


def main():
    args = docopt(__doc__)
    dropout = float(args['--dropout'])
    epochs = int(args['--epochs'])
    with_labels = args['--with_labels']
    eval_file = args['--eval']

    if args['rnn']:
        buckets = [5, 10, 15, 25, 40, -1]
        Xb, Xtb, yb, ytb, cb, ctb, vocab = get_clf_data(args['<folder>'], args['<trainpattern>'],
                                                        args['<testpattern>'], buckets)

        pkl_path = get_filename('rnn', args['<trainpattern>'])
        argdict = {'input_size': len(vocab.token_to_idx) + 1,
                   'embed_size': 128,
                   'hidden_size': 32,
                   'num_layers': 1,
                   'dropout': dropout,
                   'use_final_layer': not with_labels}
        modelfn = cluster_factory(LSTMClassifier(**argdict), 5, with_labels)

    elif args['cnn']:
        Xb, Xtb, yb, ytb, cb, ctb, vocab = get_clf_data(args['<folder>'], args['<trainpattern>'],
                                                        args['<testpattern>'], [40])

        pkl_path = get_filename('cnn', args['<trainpattern>'])
        argdict = {'input_size': len(vocab.token_to_idx) + 1,
                   'seq_len': Xb[0].shape[1],
                   'embed_size': 128,
                   'num_filters': 32,
                   'dropout': dropout,
                   'use_final_layer': not with_labels}
        modelfn = cluster_factory(CNNClassifier(**argdict), 5, with_labels)

    elif args['speaker']:
        Xb, Xtb, yb, ytb, vocab = get_speaker_data(args['<folder>'], args['<trainpattern>'],
                                                   args['<testpattern>'], 40)

        pkl_path = get_filename('speaker', args['<trainpattern>'])
        argdict = {'input_size': len(vocab.token_to_idx) + 1,
                   'seq_length': Xb[0].shape[1],
                   'embed_size': 128,
                   'encoder_hidden': 64,
                   'num_layers': 1,
                   'dropout': dropout}
        modelfn = NameClassifier

    optimfn = torch.optim.Adam

    # construct the model and optimizer
    loaded = load_model(pkl_path)
    if loaded is None:
        model = modelfn()
        optim = optimfn(model.parameters())
    else:
        model, optim = loaded

    eval_fn = lambda m: evaluate_clf(m, Xtb, ctb, ytb, silent=True) if eval_file else None
    losses, optim = train(model, optim, Xb, yb, cb, epochs, eval_fn=eval_fn)
    torch.save((modelfn, optimfn, model.state_dict(), optim.state_dict()),
               pkl_path)

    if args['rnn'] or args['cnn']:
        evaluate_clf(model, Xtb, ctb, ytb)
        write_losses(losses, eval_file)
    else:
        evaluate_spkr(model, Xtb, ytb, vocab.idx_to_token)
        write_losses(losses, 'spkr_losses.txt')


if __name__ == '__main__':
    main()
