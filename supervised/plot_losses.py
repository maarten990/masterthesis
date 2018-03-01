import sys

import fire
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')


def plot(path, label, smooth):
    with open(path, 'r') as f:
        losses = [float(l) for l in f]

    if smooth:
        losses = np.convolve(losses, [1/5] * 5, mode='valid')

    plt.plot(losses, label=label)


def main(datatype, ylabel='loss', smooth=False):
    label_path = f'{datatype}_labels.txt'
    nolabel_path = f'{datatype}_no_labels.txt'

    plot(label_path, 'with labels', smooth)
    plot(nolabel_path, 'without labels', smooth)

    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)
