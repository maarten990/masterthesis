import os.path
from lxml import etree
from glob import glob
import re
import numpy as np


class Data():
    def __init__(self, folder, pattern='*.xml'):
        self.files = list(self.__load_from_disk(folder, pattern))

    def __load_from_disk(self, folder, pattern):
        """ Load each xml file for the given folder as an etree. """
        parser = etree.XMLParser(ns_clean=True, encoding='utf-8')

        for file in glob(os.path.join(folder, pattern)):
            with open(file, 'r') as f:
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
                y.append(self.__get_label(window[round(n / 2)]))

        return np.array(X, dtype='float32'), np.array(y)


def tf_idf_featurizer(nodes, all_nodes):
    pass


def metadata_featurizer(nodes, _):
    return [node.attrib[key] for node in nodes for key in sorted(node.attrib) if key != 'is-speech']
