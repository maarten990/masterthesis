import sys
import fastcluster
import numpy as np


def main():
    data = np.loadtxt(sys.argv[1])

    print(f'Received {data.size} points, clustering...')
    clusters = fastcluster.single(data)
    print('Finished clustering')

    # from of the output: an (N-1)*4 matrix where each row is the 2 joined
    # indices along with the distance and number of points
    with open(sys.argv[2], 'w') as f:
        f.write(clusters)


if __name__ == '__main__':
    main()
