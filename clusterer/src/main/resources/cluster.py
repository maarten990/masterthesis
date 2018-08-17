import sys
import fastcluster
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    infile = sys.argv[1]
    outfile = sys.argv[2]

    data = np.genfromtxt(infile, delimiter=',')
    print('Received {} points, clustering...'.format(data.shape[0]))

    if data.size > 0:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        clusters = fastcluster.single(data)
        print('Finished clustering')
    else:
        clusters = []
        print('Insufficient data to cluster')

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
