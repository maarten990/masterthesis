import argparse
from data import create_dictionary, sliding_window, pad_sequences
from train import load_model

from tabulate import tabulate
import torch
from torch.autograd import Variable


def get_data(args, max_len=None):
    vocab = create_dictionary('./training_data', args.pattern)
    X_train, y_is_speech, Y_speaker, _ = sliding_window('./training_data', args.file,
                                                        2, 1, vocab=vocab)
    X_train = pad_sequences(X_train, max_len)
    Y_speaker = pad_sequences(Y_speaker, max_len)

    return X_train, y_is_speech, Y_speaker, vocab


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='xml file to segment')
    parser.add_argument('-p', '--pattern', default='1800*.xml',
                        help='pattern to use for dictionary generation')
    parser.add_argument('--network', '-n', choices=['rnn', 'cnn'],
                        default='rnn', help='the type of neural network to use')

    return parser.parse_args()


def zip_longest(a, b, repl=None):
    # ensure that a is larger than b
    if len(b) > len(a):
        a, b = b, a

    for i in range(len(a)):
        elem_a = a[i]
        elem_b = b[i] if i < len(b) else repl

        yield elem_a, elem_b


def main():
    args = get_args()
    X, speeches, speakers, vocab = get_data(args, 40)

    clf_path = f'pickle/clf_{args.network}.pkl'
    spkr_path = f'pickle/spkr.pkl'
    clf_model = load_model(clf_path)
    spkr_model = load_model(spkr_path)

    # classify the speeches and get the speakers for the positive classifications
    pred_is_speech = clf_model(Variable(torch.from_numpy(X).long()))
    pred_is_speech = pred_is_speech.squeeze().data.numpy()
    pred_speeches = X[pred_is_speech > 0.9, :]

    pred_speakers = spkr_model(Variable(torch.from_numpy(X[pred_is_speech > 0.9, :]).long()))
    pred_speakers = pred_speakers.data.numpy()

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
