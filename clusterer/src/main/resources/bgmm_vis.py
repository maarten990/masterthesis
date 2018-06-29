import sys
import numpy as np
from sklearn.mixture import BayesianGaussianMixture


def main():
    infile = sys.argv[1]
    outfile = sys.argv[2]
    k = int(sys.argv[3])

    data = np.genfromtxt(infile, delimiter=',')

    print('Received {} points, clustering with {} mixture components'.format(
        data.shape[0], k
    ))

    if data.size > 0:
        clusterer = BayesianGaussianMixture(k)
        clusterer.fit(data)
        labels = clusterer.predict(data)
        print('Finished clustering')
    else:
        labels = []
        print('Insufficient data to cluster')

    with open(outfile, 'w') as f:
        f.write(','.join(map(str, labels)))


if __name__ == '__main__':
    main()
