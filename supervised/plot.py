from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


def plot(curves: Dict[str, List[float]], monotonic=False) -> None:
    for label, data in curves.items():
        if monotonic:
            for i in range(1, len(data)):
                data[i] = min(data[i], data[i-1])

        plt.plot(data, label=label)

    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')


if __name__ == '__main__':
    fire.Fire(main)
