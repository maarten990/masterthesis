import sys
import numpy as np
from sklearn.cluster import DBSCAN


def main():
    infile = sys.argv[1]
    outfile = sys.argv[2]
    epsilon = float(sys.argv[3])
    min_samples = int(sys.argv[4])

    data = np.genfromtxt(infile, delimiter=',')

    print('Received {} points, clustering with eps={}, min_samples={}'.format(
        data.shape[0], epsilon, min_samples))

    if data.size > 0:
        clusterer = DBSCAN(eps=epsilon, min_samples=min_samples)
        labels = clusterer.fit_predict(data)
        print('Finished clustering')
    else:
        labels = []
        print('Insufficient data to cluster')

    with open(outfile, 'w') as f:
        f.write(','.join(map(str, labels)))


if __name__ == '__main__':
    main()
