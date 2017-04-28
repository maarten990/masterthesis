import os.path
from lxml import etree
from glob import glob
import re
import numpy as np


class Data():
    def __init__(self, folder):
        self.files = list(self.__load_from_disk(folder))

    def __load_from_disk(folder):
        """ Load each xml file for the given folder as an etree. """
        parser = etree.XMLParser(ns_clean=True, encoding='utf-8')

        for file in glob(os.path.join(folder, '*.xml')):
            with open(file, 'r') as f:
                xml = etree.fromstring(f.read().encode('utf-8'), parser)
                yield xml

    def __get_label(self, node):
        pass

    def sliding_window(self, n, featurizer):
        """
        Return a sliding window representation over the documents with the
        given feature transformation.
        """

        X = []
        y = []

        for xml in self.files:
            nodes = xml.xpath('//text')
            for node1, node2, node3 in zip(*(nodes[i:] for i in range(n))):
                X.append(featurizer([node1, node2, node3], nodes))
                y.append(self.__get_label(node2))

        return np.array(X), np.array(y)


def tf_idf_featurizer(self, nodes, all_nodes):
    pass


def metadeta_featurizer(self, nodes, _):
    return [node.attrib[key] for node in nodes for key in sorted(node.attrib)]
