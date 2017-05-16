from collections import namedtuple

from sklearn.externals import joblib
from keras.layers import Activation, Dense, Dropout, Flatten, Masking
from keras.layers import Conv1D, MaxPool1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, load_model

Split = namedtuple('Split', ['X_train', 'X_test', 'y_train', 'y_test'])


def create_cnn(timesteps, n, n_outputs=1, activation='sigmoid'):
    model = Sequential([
        Conv1D(16, 3, activation='relu', padding='same', input_shape=(timesteps, n)),
        Conv1D(16, 3, activation='relu'),
        MaxPool1D(2),
        Dropout(0.25),

        Flatten(),
        Dense(n_outputs),
        Activation(activation)
    ])

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_rnn(timesteps, n, n_outputs=1, activation='sigmoid'):
    model = Sequential([
        Masking(input_shape=(timesteps, n)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.25),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.25),
        Dense(n_outputs),
        Activation(activation)
    ])

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_neuralnet(k, dropout):
    """ Create a simple feedforward Keras neural net with k inputs """
    model = Sequential([
        Dense(100, input_dim=k, activation='relu'),
        Dropout(dropout),
        Dense(25, activation='relu'),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def save_pipeline(clf, data, name, *args):
        path = f'{name}.pkl'
        nnet = clf.model
        nnet.save(f'{name}.h5')
        clf.model = None
        joblib.dump([clf, data] + list(args), path)
        clf.model = nnet


def load_pipeline(name):
    path = f'{name}.pkl'
    args = joblib.load(path)
    args[0].model = load_model(f'{name}.h5')

    return args
