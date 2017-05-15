import os.path
import pickle
import re
from glob import glob
from math import floor

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

    def speaker_timeseries(self, n_samples, timesteps):
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

                padding = re.match(f'(.*){name}(.*)', sample)
                if not padding:
                    continue

                label = (''.join(['0' for _ in padding.groups()[0]]) + name +
                         ''.join(['0' for _ in padding.groups()[1]]))

                if label in output:
                    continue
                else:
                    input.append('0' * (timesteps - 1) + sample)
                    output.append(label)

        # do a sliding window over the inputs
        X = []
        y = []
        for seq, label in zip(input, output):
            for i in range(0, len(seq) - timesteps):
                X.append(seq[i:(i + timesteps)])
                y.append(label[i])

        # convert the text to numpy arrays
        return (np.array([[ord(ch) for ch in seq] for seq in X]),
                np.array([ord(ch) for ch in y]))


def metadata_featurizer(nodes, _):
    return [node.attrib[key] for node in nodes for key in sorted(node.attrib) if key != 'is-speech']


def char_featurizer(nodes, _):
    out = []
    for node in nodes:
        if node.text:
            out.extend([ord(c) for c in node.text])

    return out
