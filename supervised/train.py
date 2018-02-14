"""
Train the neural networks.

Usage:
train.py (rnn | cnn | speaker) <folder> <trainpattern> <testpattern>
    [--epochs=<n>] [--dropout=<ratio>] [--with-labels]
train.py (-h | --help)

Options:
    -h --help          Show this screen
    --epochs=<n>       Number of epochs to train for [default: 100]
    --dropout=<ratio>  The dropout ratio between 0 and 1 [default: 0.5]
    --with-labels      Include computed cluster labels.

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
from models import LSTMClassifier, CNNClassifier, NameClassifier

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
        modelfn, kwargs, optimfn, model_state, optim_state = torch.load(filename)
        model = modelfn(**kwargs)
        optim = optimfn(model.parameters())

        model.load_state_dict(model_state)
        optim.load_state_dict(optim_state)
        return model, optim
    else:
        return None


def write_losses(losses, basename):
    batch, epoch = losses

    with open('batch_' + basename, 'w') as f:
        f.write('\n'.join([str(l) for l in batch]))

    with open('epoch_' + basename, 'w') as f:
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
    data = sliding_window(folder, trainpattern, 2, 0.1)
    X, y = data.X, data.y
    vocab = data.vocab

    Xb, yb = pad_sequences(X, y, buckets)

    data = sliding_window(folder, testpattern, 2, 0.1, vocab=vocab)
    Xt, yt = data.X, data.y
    Xtb, ytb = pad_sequences(Xt, yt, buckets)

    for i, X in enumerate(Xb):
        train_samples = X.shape[0] if len(X) > 0 else 0
        test_samples = sum(bucket.shape[0] if len(bucket) > 0 else 0 for bucket in Xtb)
        seqlen = Xtb[i].shape[1] if len(Xtb[i]) > 0 else 0
        print('Speech classifier bucket {}: {} training samples, {} testing samples, sequence length {}'.format(
            i, train_samples, test_samples, seqlen))

    return Xb, Xtb, yb, ytb, vocab


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


def train(model, optimizer, X_buckets, y_buckets, epochs=100, batch_size=32):
    model.train()
    model.cuda()

    batch_losses = []
    epoch_losses = []
    t = trange(epochs, desc='Training')
    for _ in t:
        epoch_loss = torch.zeros(1).float()

        for X_train, y_train in zip(X_buckets, y_buckets):
            for i in range(0, X_train.shape[0], batch_size):
                X = Variable(torch.from_numpy(X_train[i:i+32, :])).long().cuda()
                y = Variable(torch.from_numpy(y_train[i:i+32])).float().cuda()

                y_pred = model(X)
                loss = model.loss(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.data.cpu())
                epoch_loss += batch_losses[-1]

        loss = epoch_loss
        epoch_losses.append(loss)
        loss_delta = epoch_losses[-1] - epoch_losses[-2] if len(epoch_losses) > 1 else [0]
        t.set_postfix({'loss': loss[0],
                       'Δloss': loss_delta[0]})

    model.eval()
    return (batch_losses, epoch_losses), optimizer


def evaluate_clf(model, Xb, yb):
    """
    Evaluate the trained model.
    Xb, yb: bucketed lists of training and test data
    """
    model = model.eval().cuda()
    predictions = []
    true = []

    for X, y in zip(Xb, yb):
        Xvar = Variable(torch.from_numpy(X)).long().cuda()
        pred = model(Xvar)
        pred = pred.cpu().squeeze().data.numpy()
        pred = np.where(pred > 0.5, 1, 0)
        predictions.extend(pred)
        true.extend(y)

    table = []
    table.append(['f1', f1_score(true, predictions)])
    table.append(['Speech recall', recall_score(true, predictions)])
    table.append(['Speech precision', precision_score(true, predictions)])

    print()
    print(tabulate(table))


def evaluate_spkr(model, Xb, yb, idx_to_token):
    model.eval()
    model.cuda()

    correct = 0
    for X, y in zip(Xb, yb):
        Xvar = Variable(torch.from_numpy(X)).long().cuda()
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

    if args['rnn']:
        buckets = [5, 10, 15, 25, 40, -1]
        Xb, Xtb, yb, ytb, vocab = get_clf_data(args['<folder>'], args['<trainpattern>'],
                                               args['<testpattern>'], buckets)

        pkl_path = get_filename('rnn', args['<trainpattern>'])
        argdict = {'input_size': len(vocab.token_to_idx) + 1,
                   'embed_size': 128,
                   'hidden_size': 32,
                   'num_layers': 1,
                   'dropout': dropout}
        modelfn = LSTMClassifier

    elif args['cnn']:
        Xb, Xtb, yb, ytb, vocab = get_clf_data(args['<folder>'], args['<trainpattern>'],
                                               args['<testpattern>'], [40])

        pkl_path = get_filename('cnn', args['<trainpattern>'])
        argdict = {'input_size': len(vocab.token_to_idx) + 1,
                   'seq_len': Xb[0].shape[1],
                   'embed_size': 128,
                   'num_filters': 32,
                   'dropout': dropout}
        modelfn = CNNClassifier

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
        model = modelfn(**argdict)
        optim = optimfn(model.parameters())
    else:
        model, optim = loaded

    losses, optim = train(model, optim, Xb, yb, epochs)
    torch.save((modelfn, argdict, optimfn, model.state_dict(), optim.state_dict()),
               pkl_path)

    if args['rnn'] or args['cnn']:
        evaluate_clf(model, Xtb, ytb)
        write_losses(losses, 'clf_losses.txt')
    else:
        evaluate_spkr(model, Xtb, ytb, vocab.idx_to_token)
        write_losses(losses, 'spkr_losses.txt')


if __name__ == '__main__':
    main()
