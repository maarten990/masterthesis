import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import BayesianGaussianMixture


def prune(clusterer, labels, min_weight):
    samples, _ = clusterer.sample(1000)
    gen_labels = clusterer.predict_proba(samples)
    weights = gen_labels.mean(axis=0)

    significant = weights >= min_weight
    out = labels[:, significant]

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_w = np.arange(np.shape(out)[1]) + 1
    ax.bar(plot_w - 0.5, np.mean(out, axis=0), width=1., lw=0)

    ax.set_xlim(0.5, np.shape(out)[1])
    ax.set_xlabel("Component")
    ax.set_ylabel("Posterior expected mixture weight")
    plt.savefig("component_dist.png")

    print(
        f"Reduced number of clusters from {np.shape(labels)[1]} to {np.shape(out)[1]}"
    )
    return out


def main():
    infile = sys.argv[1]
    outfile = sys.argv[2]
    k = int(sys.argv[3])
    wp = int(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    data = np.genfromtxt(infile, delimiter=',')

    print(
        "Received {} points, clustering with {} mixture components and 10 inits".format(
            data.shape[0], k
        )
    )

    if data.size > 0:
        clusterer = BayesianGaussianMixture(
            k, n_init=10, weight_concentration_prior=wp
        )
        clusterer.fit(data)
        labels = clusterer.predict_proba(data)
        labels = prune(clusterer, labels, 0.01)
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
