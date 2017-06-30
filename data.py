import os.path
import pickle
import random
import re
from glob import glob

import nltk
import numpy as np
from lxml import etree

XMLNS = {'pm': 'http://www.politicalmashup.nl',
         'dc': 'http://purl.org/dc/elements/1.1'}


class Vocab(object):
    def __init__(self, token_to_idx, idx_to_token):
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token


def pickler(func):
    def wrapped(*args, **kwargs):
        args_str = '_'.join(str(arg) for arg in args)
        kwargs_str = '_'.join(f'{key}_{val}' for key, val in kwargs)

        # remove any illegal/stupid characters
        args_str = re.sub(r'[/\\\*]', '', args_str)
        kwargs_str = re.sub(r'[/\\\*]', '', kwargs_str)

        pkl_path = os.path.join('pickle', f'{func.__name__}_{args_str}_{kwargs_str}.pkl')

        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        else:
            out = func(*args, **kwargs)
            with open(pkl_path, 'wb') as f:
                pickle.dump(out, f)

            return out

    return wrapped


def load_from_disk(folder, pattern):
    """ Load each xml file for the given folder as an etree. """
    parser = etree.XMLParser(ns_clean=True, encoding='utf-8')

    for file in glob(os.path.join(folder, pattern)):
        with open(file, 'r', encoding='utf-8') as f:
            xml = etree.fromstring(f.read().encode('utf-8'), parser)
            yield xml


def get_label(node):
    is_speech = node.attrib['is-speech']
    return 1 if is_speech == 'true' else 0


@pickler
def create_dictionary(raw_folder, pattern):
    all_words = set()
    for i, xml in enumerate(load_from_disk(raw_folder, pattern)):
        print(i)
        text = ' '.join(xml.xpath('//text//text()')).lower()
        tokens = nltk.tokenize.word_tokenize(text)
        all_words |= set(tokens)

    word_to_idx = {w: i+1 for i, w in enumerate(all_words)}
    idx_to_word = {i+1: w for i, w in enumerate(all_words)}

    return Vocab(word_to_idx, idx_to_word)


def sliding_window(raw_folder, pattern, n, prune_ratio, label_pos=0, vocab=None):
    """
    Return a sliding window representation over the documents with the
    given feature transformation.
    """
    if not vocab:
        vocab = create_dictionary(raw_folder, pattern)
    
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    Xs = []
    ys = []

    for xml in load_from_disk(raw_folder, pattern):
        # First get all text nodes.
        nodes = xml.xpath('//text')

        # Then tokenize the entire corpus, while keeping track of the indices that
        # correspond to the boundaries between the nodes as well as the labels.
        tokens = []
        boundaries = []
        labels = []
        current_idx = 0
        for node in nodes:
            boundaries.append(current_idx)
            text = ' '.join(node.xpath('.//text()'))
            node_tokens = [token.lower() for token in tokenizer.tokenize(text)]

            tokens.extend(node_tokens)
            labels.append(get_label(node))
            current_idx += len(node_tokens)

        # Finally, iterate over the windows to construct the training data.
        # Only iterate until the -n'th element to account for the window size.
        for idx in range(len(boundaries[:-n])):
            window_tokens = []
            for offset in range(n):
                start = boundaries[idx + offset]
                end = boundaries[idx + offset + 1]
                window_tokens.extend(tokens[start:end])

            y = labels[idx + label_pos]

            # skip empty lines
            if len(window_tokens) == 0:
                continue

            # random pruning for negative labels
            if y == 0 and random.random() > prune_ratio:
                continue

            Xs.append([vocab.token_to_idx[token] if token in vocab.token_to_idx else 0
                       for token in window_tokens])
            ys.append(y)

    return Xs, np.array(ys), vocab


def speaker_timeseries(parsed_folder, pattern):
    input = []
    output = []
    seen_names = set()
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    for xml in load_from_disk(parsed_folder, pattern):
        for speech in xml.xpath('//pm:speech', namespaces=XMLNS):
            name = speech.xpath('./@pm:speaker', namespaces=XMLNS)[0]
            function = speech.xpath('./@pm:function', namespaces=XMLNS)[0]

            if name in seen_names:
                continue
            else:
                seen_names.add(name)

            if function == 'De Duitser':
                party = speech.xpath('./@pm:party', namespaces=XMLNS)[0]
                sample = '{} ({})'.format(name, party)
            elif 'sident' in function:
                sample = '{} {}'.format(function, name)
            elif 'inister' in function:
                sample = '{}, {}'.format(name, function)
            else:
                continue

            tokens = tokenizer.tokenize(sample)
            name_tokens = tokenizer.tokenize(name)

            input.append(tokens)
            output.append([1 if token in name_tokens else 0 for token in tokens])

    all_words = set([word for sample in input for word in sample])
    word_to_idx = {w: i+1 for i, w in enumerate(all_words)}
    idx_to_word = {i+1: w for i, w in enumerate(all_words)}

    X = [[word_to_idx[w] for w in sample]
         for sample in input]
    Y = output

    X_out = pad_lists(X)
    Y_out = pad_lists(Y)

    return X_out, Y_out, word_to_idx, idx_to_word


def token_featurizer(nodes):
    out = []
    for node in nodes:
        text = ' '.join(node.xpath('.//text()'))
        tokens = nltk.tokenize.word_tokenize(text)
        out.extend(tokens)

    return [token.lower() for token in out]


def pad_lists(lists, max_width=None):
    if not max_width:
        max_width = max(map(len, lists))

    out = np.zeros((len(lists), max_width))

    for i, l in enumerate(lists):
        out[i, :] = np.pad(l, (0, max_width - len(l)), 'constant')

    return out


def sentences_to_input(sentences, char_to_idx, max_length):
    X = [nltk.tokenize.word_tokenize(sent)
         for sent in sentences]

    X = [[char_to_idx[w] if w in char_to_idx else -3 for w in sent] for sent in X]

    return pad_lists(X, max_length)
