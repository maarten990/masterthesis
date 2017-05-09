import os.path
from glob import glob
from math import floor

import numpy as np
from lxml import etree


class Data():
    def __init__(self, folder, pattern='*.xml'):
        self.files = list(self.__load_from_disk(folder, pattern))

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

        X = []
        y = []

        for xml in self.files:
            nodes = xml.xpath('//text')
            for window in zip(*(nodes[i:] for i in range(n))):
                X.append(featurizer(window, nodes))
                y.append(self.__get_label(window[floor(n / 2)]))

        return X, np.array(y)


def metadata_featurizer(nodes, _):
    return [node.attrib[key] for node in nodes for key in sorted(node.attrib) if key != 'is-speech']


def char_featurizer(nodes, _):
    out = []
    for node in nodes:
        if node.text:
            out.extend([ord(c) for c in node.text])

    return out
