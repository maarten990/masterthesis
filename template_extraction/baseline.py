import sys
from collections import Counter

import nltk


def get_document_frequencies(documents):
    doc_freq = Counter()
    for doc in documents:
        for token in set(doc):
            doc_freq[token] += 1

    return doc_freq


def extract_phrases(documents, gamma=1):
    """
    documents: list of tokenized documents
    gamma: ratio of documents required to contain a phrase
    """
    N = len(documents)
    doc_freq = get_document_frequencies(documents)
    acc = Counter()

    for D in documents:
        phrase = []
        for token in D:
            if doc_freq[token] >= gamma * N:
                phrase.append(token)
            else:
                if len(phrase) > 0:
                    acc[tuple(phrase)] += 1
                phrase = []

        if len(phrase) > 0:
            acc[tuple(phrase)] += 1

    return [phrase for phrase in acc if acc[phrase] >= gamma * N]


def main():
    documents = []
    for path in sys.argv[1:]:
        with open(path, 'r') as f:
            documents.append(nltk.tokenize.word_tokenize(f.read()))

    for phrase in extract_phrases(documents, 0.9):
        print(' '.join(phrase))


if __name__ == '__main__':
    main()
