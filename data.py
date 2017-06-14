import os.path
import pickle
from glob import glob
from math import floor

import nltk
import numpy as np
from lxml import etree

XMLNS = {'pm': 'http://www.politicalmashup.nl',
         'dc': 'http://purl.org/dc/elements/1.1'}


class Data():
    def __init__(self, raw_folder, parsed_folder, pattern='*.xml'):
        self.raw = list(self.__load_from_disk(raw_folder, pattern))
        self.parsed = list(self.__load_from_disk(parsed_folder, pattern))

    def __load_from_disk(self, folder, pattern):
        """ Load each xml file for the given folder as an etree. """
        parser = etree.XMLParser(ns_clean=True, encoding='utf-8')

        for file in glob(os.path.join(folder, pattern)):
            with open(file, 'r', encoding='utf-8') as f:
                xml = etree.fromstring(f.read().encode('utf-8'), parser)
                yield xml

    def __get_label(self, node):
        is_speech = node.attrib['is-speech']
        return 1 if is_speech == 'true' else 0

    def sliding_window(self, n, featurizer):
        """
        Return a sliding window representation over the documents with the
        given feature transformation.
        """
        pkl_path = f'pickle/sliding_{n}_{featurizer.__name__}.pkl'

        # try to load from disk
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)

        X = []
        y = []

        for xml in self.raw:
            nodes = xml.xpath('//text')
            for window in zip(*(nodes[i:] for i in range(n))):
                X.append(featurizer(window, nodes))
                y.append(self.__get_label(window[floor(n / 2)]))

        # save to disk before returning
        with open(pkl_path, 'wb') as f:
            pickle.dump((X, np.array(y)), f)

        return X, np.array(y)

    def speaker_timeseries(self):
        input = []
        output = []

        for xml in self.parsed:
            for speech in xml.xpath('//pm:speech', namespaces=XMLNS):
                name = speech.xpath('./@pm:speaker', namespaces=XMLNS)[0]
                function = speech.xpath('./@pm:function', namespaces=XMLNS)[0]

                if function == 'De Duister':
                    party = speech.xpath('./@pm:party', namespaces=XMLNS)[0]
                    sample = f'{name} ({party})'
                elif 'sident' in function:
                    sample = f'{function} {name}'
                elif 'inister' in function:
                    sample = f'{name}, {function}'
                else:
                    continue

                tokens = nltk.tokenize.word_tokenize(sample) + ['<END>']
                input.append(tokens)

                output.append([-1]
                              + [input[-1].index(w) for w in nltk.tokenize.word_tokenize(name)]
                              + [-2])

        all_words = set([word for sample in input for word in sample])
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        idx_to_word = {i: w for i, w in enumerate(all_words)}

        X = [[word_to_idx[w] for w in sample]
             for sample in input]
        Y = output

        X_out = pad_lists(X)
        Y_out = pad_lists(Y)

        return X_out, Y_out, word_to_idx, idx_to_word


def metadata_featurizer(nodes, _):
    return [node.attrib[key] for node in nodes for key in sorted(node.attrib) if key != 'is-speech']


def char_featurizer(nodes, _):
    out = []
    for node in nodes:
        if node.text:
            out.extend([ord(c) for c in node.text])

    return out


def pad_lists(lists):
    max_width = max(map(len, lists))
    out = np.zeros((len(lists), max_width))

    for i, l in enumerate(lists):
        out[i, :] = np.pad(l, (0, max_width - len(l)), 'constant')

    return out