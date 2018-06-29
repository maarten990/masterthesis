import sys
import numpy as np
from sklearn.mixture import GaussianMixture


def main():
    infile = sys.argv[1]
    outfile = sys.argv[2]
    k = int(sys.argv[3])

    data = np.genfromtxt(infile, delimiter=',')

    print('Received {} points, clustering with {} mixture components'.format(
        data.shape[0], k
    ))

    if data.size > 0:
        clusterer = GaussianMixture(k)
        clusterer.fit(data)
        labels = clusterer.predict_proba(data)
        print('Finished clustering')
    else:
        labels = []
        print('Insufficient data to cluster')

    with open(outfile, 'w') as f:
        for sample in labels:
            f.write(", ".join(map(str, sample)))
            f.write("\n")


if __name__ == '__main__':
    main()
