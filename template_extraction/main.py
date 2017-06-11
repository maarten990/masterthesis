import sys
from collections import OrderedDict
from suffixarray import SuffixArray
from nltk.tokenize import word_tokenize


def phrase(start, length, text):
    return ' '.join(text[start:(start+length)])


def extract_phrases(sa, text, gamma, mu):
    """
    sa: a suffix array
    lcp: the corresponding longest common prefix array
    gamma: eh
    mu: minimum number of shared prefix tokens
    """
    acc = OrderedDict()
    acc_lcp = OrderedDict()
    
    acc[phrase(sa.array[0], sa.lcp[0], text)] = 1
    acc_lcp[phrase(sa.array[0], sa.lcp[0], text)] = 0

    for i in range(1, len(sa.array)):
        if sa.lcp[i] > mu:
            acc[phrase(sa.array[i], sa.lcp[i - 1], text)] = 1
            acc_lcp[phrase(sa.array[i], sa.lcp[i - 1], text)] = sa.lcp[i-1]

            if sa.lcp[i] < sa.lcp[i - 1]:
                c = acc[phrase(sa.array[i], sa.lcp[i - 1], text)]
                acc[phrase(sa.array[i], sa.lcp[i], text)] = c
                acc_lcp[phrase(sa.array[i], sa.lcp[i], text)] = sa.lcp[i]

                prev_larger = [p for p, cp in acc_lcp.items() if cp > sa.lcp[i]]
                for p in prev_larger:
                    if p in acc.keys():
                        if acc[p] < gamma * 1:
                            del acc[p]
                            del acc_lcp[p]
                    else:
                        print(f'Warning: {p} in acc_lcp but not in acc')

            elif sa.lcp[i] > sa.lcp[i - 1]:
                acc[phrase(sa.array[i], sa.lcp[i], text)] = 1
                acc_lcp[phrase(sa.array[i], sa.lcp[i], text)] = sa.lcp[i]
                prev_smaller = (p for p, cp in acc_lcp.items() if cp < sa.lcp[i])
                for p in prev_smaller:
                    acc[phrase(sa.array[i], sa.lcp[i], text)] += 1
                    acc_lcp[phrase(sa.array[i], sa.lcp[i], text)] += sa.lcp[i]

    return list(acc.keys())


def main():
    text = []
    for path in sys.argv[1:]:
        with open(path, 'r', encoding='utf-8') as f:
            text += word_tokenize(f.read() + '$')

    sa = SuffixArray(text)
    phrases = extract_phrases(sa, text, 1, 4)
    import IPython
    IPython.embed()


if __name__ == '__main__':
    main()
