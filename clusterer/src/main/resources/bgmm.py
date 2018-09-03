import sys
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler


def prune(clusterer, labels, min_weight):
    samples, _ = clusterer.sample(1000)
    gen_labels = clusterer.predict_proba(samples)
    weights = gen_labels.mean(axis=0)

    significant = weights >= min_weight
    out = labels[:, significant]

    print(
        f"Reduced number of clusters from {np.shape(labels)[1]} to {np.shape(out)[1]}"
    )
    return out


def main():
    infile = sys.argv[1]
    outfile = sys.argv[2]
    k = int(sys.argv[3])

    data = np.genfromtxt(infile, delimiter=',')

    print(
        "Received {} points, clustering with {} mixture components and 2 inits".format(
            data.shape[0], k
        )
    )

    if data.size > 0:
        scaler = StandardScaler()
        clusterer = BayesianGaussianMixture(
            k,
            n_init=2,
        )

        data = scaler.fit_transform(data)

        converged = False
        while not converged:
            try:
                clusterer.fit(data)
                converged = True
            except ValueError:
                clusterer.n_components -= 1
                print(f"Retrying with {clusterer.n_components} components.")

        labels = clusterer.predict_proba(data)
        labels = prune(clusterer, labels, 0.001)
        print("Finished clustering")
    else:
        labels = []
        print("Insufficient data to cluster")

    with open(outfile, 'w') as f:
        for sample in labels:
            f.write(", ".join(map(str, sample)))
            f.write("\n")


if __name__ == '__main__':
    main()
