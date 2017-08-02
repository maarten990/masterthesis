"""
Segment an xml file into speeches.

Usage:
segment.py (rnn | cnn) <file> [--dictpattern=<p>]
segment.py (-h | --help)

Options:
    -h --help          Show this screen
    --dictpattern=<p>  Glob pattern to create the dictionary [default: 1800*.xml]
"""

import os.path
from data import create_dictionary, sliding_window, pad_sequences
from docopt import docopt
from train import load_model, get_filename

import numpy as np
from tabulate import tabulate
import torch
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch.autograd import Variable


def get_data(filepath, dictpattern, buckets):
    vocab = create_dictionary('training_data', dictpattern)

    folder, fname = os.path.split(filepath)
    X_train, y_is_speech, Y_speaker, _ = sliding_window(folder, fname, 2, 1,
                                                        vocab=vocab)
    Xb, yb = pad_sequences(X_train, y_is_speech, buckets)

    return Xb, yb, vocab


def main():
    args = docopt(__doc__)
    buckets = [40]
    Xb, yb, vocab = get_data(args['<file>'], args['--dictpattern'], buckets)

    # unbucket
    X = Xb[0]
    speeches = yb[0]

    clftype = 'cnn' if args['cnn'] else 'rnn'

    clf_path = get_filename(clftype, args['--dictpattern'])
    spkr_path = get_filename('speaker', args['--dictpattern'])
    clf_model, _ = load_model(clf_path)
    spkr_model, _ = load_model(spkr_path)
    clf_model.eval()
    spkr_model.eval()

    # classify the speeches and get the speakers for the positive classifications
    pred_is_speech = clf_model(Variable(torch.from_numpy(X).long()))
    pred_is_speech = pred_is_speech.squeeze().data.numpy()
    pred_speeches = X[pred_is_speech > 0.5, :]

    pred_speakers = spkr_model(Variable(torch.from_numpy(X[pred_is_speech > 0.5, :]).long()))
    pred_speakers = pred_speakers.data.numpy()

    pred_speeches_binary = np.where(pred_is_speech > 0.5, 1, 0)
    print('---')
    print(f'Precision: {precision_score(speeches, pred_speeches_binary)}')
    print(f'Recall: {recall_score(speeches, pred_speeches_binary)}')
    print('---')

    # print all the predicted speeches
    speech_table = []
    for speech, speaker in zip(pred_speeches, pred_speakers):
        speech_words = [vocab.idx_to_token[i] for i in speech if i != 0]
        speaker_indices = speech[speaker > 0.5]
        speaker_words = [vocab.idx_to_token[i] for i in speaker_indices if i != 0]

        speech_table.append([' '.join(speech_words), ' '.join(speaker_words)])

    print(tabulate(speech_table, headers=['speech', 'speaker']))


if __name__ == '__main__':
    main()
