import sys
import numpy as np
from sklearn.cluster import KMeans


def main():
    data = np.genfromtxt(sys.argv[1], delimiter=',')

    print('Received {} points, clustering...'.format(data.size))

    print('Finished clustering')

    # from of the output:
    # indices along with the distance and number of points
    with open(sys.argv[2], 'w') as f:
        pass


if __name__ == '__main__':
    main()
