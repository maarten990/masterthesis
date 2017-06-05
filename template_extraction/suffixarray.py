from collections import defaultdict
from itertools import takewhile


class SuffixArray(object):
    def __init__(self, tokens):
        self.tokens = tokens
        self.array = bucketsort(tokens, range(len(tokens)), 1)
        self.lcp = [0 for _ in self.array]

        for idx in range(1, len(self.array)):
            seq_prev = tokens[self.array[idx - 1]:]
            seq_cur = tokens[self.array[idx]:]

            common_prefix = takewhile(lambda x: x[0] == x[1],
                                      zip(seq_prev, seq_cur))

            self.lcp[idx] = len(list(common_prefix))

    def longest_prefix(self):
        idx = max(range(len(self.array)), key=lambda x: self.lcp[x])
        start = self.array[idx]
        n = self.lcp[idx]

        return self.tokens[start:(start+n)]

def bucketsort(tokens, bucket, num_lookahead):
    d = defaultdict(list)

    for i in bucket:
        key = tuple(tokens[i:(i+num_lookahead)])
        d[key].append(i)

    result = []
    for k, v in sorted(d.items()):
        if len(v) > 1:
            result += bucketsort(tokens, v, num_lookahead * 2)
        else:
            result.append(v[0])

    return result
