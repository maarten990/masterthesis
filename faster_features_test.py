import data
import random
import timeit
import numpy as np


def old_sliding_window(raw_folder, pattern, n, prune_ratio, label_pos=0, vocab=None):
    """
    Return a sliding window representation over the documents with the
    given feature transformation.
    """
    if not vocab:
        vocab = data.create_dictionary(raw_folder, pattern)

    Xs = []
    ys = []
    for xml in data.load_from_disk(raw_folder, pattern):
        nodes = xml.xpath('//text')
        for window in zip(*(nodes[i:] for i in range(n))):
            tokens = data.token_featurizer(window)
            y = data.get_label(window[label_pos])

            # skip empty lines
            if len(tokens) == 0:
                continue

            # random pruning for negative labels
            if y == 0 and random.random() > prune_ratio:
                continue

            Xs.append([vocab.token_to_idx[token] if token in vocab.token_to_idx else 0
                       for token in tokens])
            ys.append(y)

    return Xs, np.array(ys), vocab


start = timeit.default_timer()
X_old, y_old, _ = old_sliding_window('raw_data/', '1800*.xml', 2, 0.1)
end_old = timeit.default_timer()
X, y, _ = data.sliding_window('raw_data/', '1800*.xml', 2, 0.1)
end = timeit.default_timer()

old_time = end_old - start
new_time = end - end_old

print(f'Old function: {old_time} seconds')
print(f'New function: {new_time} seconds')
