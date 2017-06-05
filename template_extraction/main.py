import sys
from suffixarray import SuffixArray
from nltk.tokenize import word_tokenize


def main():
    text = []
    for path in sys.argv[1:]:
        with open(path, 'r') as f:
            text += word_tokenize(f.read() + '$')

    sa = SuffixArray(text)
    print(' '.join(sa.longest_prefix()))


if __name__ == '__main__':
    main()
