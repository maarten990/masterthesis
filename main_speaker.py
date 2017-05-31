import argparse
from collections import namedtuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data import Data

Split = namedtuple('Split', ['X_train', 'X_test', 'y_train', 'y_test'])


def build_recurrent_model(seq_length, input_dim, output_dim, num_hidden):
    X = tf.placeholder(tf.float32, shape=[None, seq_length, input_dim])
    labels = tf.placeholder(tf.float32, shape=[None, seq_length, output_dim])

    # encode
    X_list = tf.unstack(X, axis=1)
    encoder = tf.contrib.rnn.LSTMCell(num_hidden, use_peepholes=False)
    states, h = tf.contrib.rnn.static_rnn(encoder, X_list,
                                          dtype=tf.float32)
    # attention weights
    # the i'th row is the weights for the i'th output
    attn_matrix = tf.Variable(tf.random_normal([seq_length, seq_length], stddev=0.1))

    # decode
    decoder = tf.contrib.rnn.LSTMCell(num_hidden, use_peepholes=False)
    W_out = tf.Variable(tf.random_normal([num_hidden, output_dim], stddev=0.1))

    predictions = []
    last_pred = tf.zeros_like(X_list[0])
    for i in range(seq_length):
        attn_weights = tf.nn.softmax(attn_matrix[i, :])
        weighted_state = attn_weights[0] * states[0]
        for j in range(1, len(states)):
            weighted_state += attn_weights[j] * states[j]

        out, _ = decoder(last_pred, (weighted_state, weighted_state))
        pred = tf.matmul(out, W_out)
        last_pred = pred
        predictions.append(pred)

    # calculate the loss
    labels_idx = tf.argmax(labels, axis=2)
    logits = tf.stack(predictions, axis=1)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(logits, [-1, output_dim]),
        labels=tf.reshape(labels_idx, [-1]))

    loss = tf.reduce_sum(loss)

    return (X, labels,
            tf.map_fn(lambda e: tf.argmax(e, axis=1), predictions, dtype=tf.int64),
            loss)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder containing the training data')
    parser.add_argument('parsed_folder', help='folder containing the parsed data')
    parser.add_argument('-t', '--test_size', type=float, default=0.25,
                        help='ratio of testing date')
    parser.add_argument('-p', '--pattern', default='*.xml',
                        help='pattern to match in the training folder')
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='number of epochs to train for')
    parser.add_argument('--load_from_disk', '-l', action='store_true',
                        help='load previously trained model from disk')
    parser.add_argument('--steps', type=int, default=15,
                        help='number of timesteps to use per sequence')

    return parser.parse_args()


def get_data(args):
    data = Data(args.folder, args.parsed_folder, args.pattern)

    X, y, char_to_idx, idx_to_char = data.speaker_timeseries()
    # X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    # y = np.reshape(y, (y.shape[0], y.shape[1], 1))

    split = Split(*train_test_split(X, y, test_size=args.test_size))

    print('{} training samples, {} testing samples'.format(
        split.X_train.shape[0], split.X_test.shape[0]))
    print('Number of features: {}'.format(split.X_train.shape[1]))

    return split, char_to_idx, idx_to_char


def extract_speaker(model, sentence, timesteps, char_to_idx, idx_to_char):
    seq = '0' * (timesteps - 1) + sentence

    X = []
    for i in range(0, len(seq) - timesteps):
            X.append(seq[i:(i + timesteps)])

    X_in = np.zeros((len(X), timesteps, len(char_to_idx)))

    for i in range(X_in.shape[0]):
        for j in range(X_in.shape[1]):
            X_in[i, j, char_to_idx[X[i][j]]] = 1

    out = ''.join(idx_to_char[i] for i in model.predict(X_in))

    return out.replace('0', '')


def main():
    args = get_args()

    split, char_to_idx, idx_to_char = get_data(args)
    print('a')
    X, y, output, loss = build_recurrent_model(split.X_train.shape[1],
                                               len(char_to_idx),
                                               len(char_to_idx), 64)
    print('b')

    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)
    init_vars = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_vars)

        for epoch in range(args.epochs):
            epoch_loss = 0
            for i in range(0, split.X_train.shape[0], 32):
                l = sess.run(loss, {X: split.X_train[i:i + 32, :, :],
                                    y: split.y_train[i:i + 32, :, :]})
                sess.run(train, {X: split.X_train[i:i + 32, :, :],
                                 y: split.y_train[i:i + 32, :, :]})

                epoch_loss += l

            print('Epoch {}: loss {:.2f}'.format(
                epoch, epoch_loss / split.X_train.shape[0]))

    # tests = ['Macaroni Sklepi, Bundesminister fur Winkels',
    #          'Marimba Wokkels (Die Linke)']


if __name__ == '__main__':
    main()
