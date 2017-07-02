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


def get_speaker(node):
    is_speech = node.attrib['is-speech']

    if is_speech == 'true':
        return node.attrib['speaker']    
    else:
        return ''


@pickler
def create_dictionary(folder, pattern):
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    all_words = set()

    for i, xml in enumerate(load_from_disk(folder, pattern)):
        print(i)
        text = ' '.join(xml.xpath('//text//text()')).lower()
        tokens = tokenizer.tokenize(text)
        all_words |= set(tokens)

    word_to_idx = {w: i+1 for i, w in enumerate(all_words)}
    idx_to_word = {i+1: w for i, w in enumerate(all_words)}

    return Vocab(word_to_idx, idx_to_word)


def sliding_window(folder, pattern, n, prune_ratio, label_pos=0, vocab=None):
    """
    Return a sliding window representation over the documents with the
    given feature transformation.
    """
    if not vocab:
        vocab = create_dictionary(folder, pattern)
    
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    inputs = []
    is_speech = []
    speakers = []

    for xml in load_from_disk(folder, pattern):
        nodes = xml.xpath('//text')
        for window in zip(*(nodes[i:] for i in range(n))):
            tokens = token_featurizer(window, tokenizer)
            y = get_label(window[label_pos])
            speaker_tokens = tokenizer.tokenize(get_speaker(window[label_pos]))

            # skip empty lines
            if len(tokens) == 0:
                continue

            # random pruning for negative labels
            if y == 0 and random.random() > prune_ratio:
                continue

            inputs.append([vocab.token_to_idx[token] if token in vocab.token_to_idx else 0
                           for token in tokens])
            is_speech.append(y)
            speakers.append([1 if token in speaker_tokens else 0 for token in tokens])

    return inputs, np.array(is_speech), speakers, vocab


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


def token_featurizer(nodes, tokenizer):
    out = []
    for node in nodes:
        text = ' '.join(node.xpath('.//text()'))
        tokens = tokenizer.tokenize(text)
        out.extend(tokens)

    return [token.lower() for token in out]


def pad_sequences(X, max_len=None):
    if not max_len:
        max_len = max(len(seq) for seq in X)

    padded = []
    for seq in X:
        diff = max_len - len(seq)
        if diff > 0:
            padded.append(np.pad(seq, (0, max_len - len(seq)), 'constant'))
        else:
            padded.append(seq[:max_len])

    return np.array(padded)




def sentences_to_input(sentences, char_to_idx, max_length):
    X = [nltk.tokenize.word_tokenize(sent)
         for sent in sentences]

    X = [[char_to_idx[w] if w in char_to_idx else -3 for w in sent] for sent in X]

    return pad_lists(X, max_length)
