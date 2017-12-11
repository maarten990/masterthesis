import sys
import numpy as np
from sklearn.cluster import KMeans


def main():
    infile = sys.argv[1]
    outfile = sys.argv[2]
    k = int(sys.argv[3])

    data = np.genfromtxt(infile, delimiter=',')

    print('Received {} points, clustering with k={}'.format(data.shape[0], k))

    clusterer = KMeans(n_clusters=k)
    labels = clusterer.fit_predict(data)

    print('Finished clustering')

    with open(outfile, 'w') as f:
        f.write(','.join(map(str, labels)))


if __name__ == '__main__':
    main()
