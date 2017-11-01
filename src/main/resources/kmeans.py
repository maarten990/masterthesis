import sys
import numpy as np
from sklearn.cluster import KMeans


def main():
    data = np.genfromtxt(sys.argv[1], delimiter=',').reshape(-1, 1)

    print('Received {} points, clustering...'.format(data.shape[0]))

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)
    clusters = kmeans.cluster_centers_.reshape(-1)

    print('Finished clustering')

    with open(sys.argv[2], 'w') as f:
        f.write(','.join(str(centroid) for centroid in clusters))


if __name__ == '__main__':
    main()
