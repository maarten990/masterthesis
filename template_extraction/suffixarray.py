from collections import defaultdict

class SuffixArray(object):
    def __init__(self, string):
        self.array = self.__bucketsort(string, range(len(string)), 1)

    def __bucketsort(self, string, bucket, order):
        d = defaultdict(list)

        for i in bucket:
            key = string[i:(i+order)]
            d[key].append(i)

        result = []
        for k, v in sorted(d.items()):
            if len(v) > 1:
                result += self.__bucketsort(string, v, order * 2)
            else:
                result.append(v[0])

        return result