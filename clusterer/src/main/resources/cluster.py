import sys
import fastcluster
import numpy as np


def main():
    data = np.genfromtxt(sys.argv[1], delimiter=',')

    print('Received {} points, clustering...'.format(data.shape[0]))
    clusters = fastcluster.single(data)
    print('Finished clustering')

    # from of the output: an (N-1)*4 matrix where each row is the 2 joined
    # indices along with the distance and number of points
    with open(sys.argv[2], 'w') as f:
        string_repr = ''
        for row in clusters:
            string_repr += ','.join(map(str, row))
            string_repr += '\n'

        f.write(string_repr)


if __name__ == '__main__':
    main()
