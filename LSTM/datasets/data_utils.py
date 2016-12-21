'''
Created on Dec 20, 2016

@author: jixiang
'''
from __future__ import division, print_function, absolute_import

import os
import random
import numpy as np
from PIL import Image
import pickle
import csv

"""
Preprocessing provides some useful functions to preprocess data before
training, such as pictures dataset building, sequence padding, etc...
Note: Those preprocessing functions are only meant to be directly applied to
data, they are not meant to be use with Tensors or Layers.
"""

_EPSILON = 1e-8


# =======================
# TARGETS (LABELS) UTILS
# =======================


def to_categorical(y, nb_classes):
    """ to_categorical.
    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    Arguments:
        y: `array`. Class vector to convert.
        nb_classes: `int`. Total number of classes.
    """
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


# =====================
#    SEQUENCES UTILS
# =====================


def pad_sequences(sequences, maxlen=None, dtype='str', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences.
    Pad each sequence to the same length: the length of the longest sequence.
    If maxlen is provided, any sequence longer than maxlen is truncated to
    maxlen. Truncation happens off either the beginning or the end (default)
    of the sequence. Supports pre-padding and post-padding (default).
    Arguments:
        sequences: list of lists where each element is a sequence.
        maxlen: int, maximum length.
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    Returns:
        x: `numpy array` with dimensions (number_of_sequences, maxlen)
    Credits: From Keras `pad_sequences` function.
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def string_to_semi_redundant_sequences(string, seq_maxlen=25, redun_step=3, char_idx=None):
    """ string_to_semi_redundant_sequences.
    Vectorize a string and returns parsed sequences and targets, along with
    the associated dictionary.
    Arguments:
        string: `str`. Lower-case text from input text file.
        seq_maxlen: `int`. Maximum length of a sequence. Default: 25.
        redun_step: `int`. Redundancy step. Default: 3.
        char_idx: 'dict'. A dictionary to convert chars to positions. Will be automatically generated if None
    Returns:
        A tuple: (inputs, targets, dictionary)
    """

    print("Vectorizing text...")

    if char_idx is None:
      char_idx = chars_to_dictionary(string)

    len_chars = len(char_idx)

    sequences = []
    next_chars = []
    for i in range(0, len(string) - seq_maxlen, redun_step):
        sequences.append(string[i: i + seq_maxlen])
        next_chars.append(string[i + seq_maxlen])

    X = np.zeros((len(sequences), seq_maxlen, len_chars), dtype=np.bool)
    Y = np.zeros((len(sequences), len_chars), dtype=np.bool)
    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t, char_idx[char]] = 1
        Y[i, char_idx[next_chars[i]]] = 1

    print("Text total length: {:,}".format(len(string)))
    print("Distinct chars   : {:,}".format(len_chars))
    print("Total sequences  : {:,}".format(len(sequences)))

    return X, Y, char_idx


def textfile_to_semi_redundant_sequences(path, seq_maxlen=25, redun_step=3,
                                         to_lower_case=False, pre_defined_char_idx=None):
    """ Vectorize Text file """
    text = open(path).read()
    if to_lower_case:
        text = text.lower()
    return string_to_semi_redundant_sequences(text, seq_maxlen, redun_step, pre_defined_char_idx)


def chars_to_dictionary(string):
    """ Creates a dictionary char:integer for each unique character """
    chars = set(string)
    # sorted tries to keep a consistent dictionary, if you run a second time for the same char set
    char_idx = {c: i for i, c in enumerate(sorted(chars))}
    return char_idx


def random_sequence_from_string(string, seq_maxlen):
    rand_index = random.randint(0, len(string) - seq_maxlen - 1)
    return string[rand_index: rand_index + seq_maxlen]


def random_sequence_from_textfile(path, seq_maxlen):
    text = open(path).read()
    return random_sequence_from_string(text, seq_maxlen)

try:
    from tensorflow.contrib.learn.python.learn.preprocessing.text import \
        VocabularyProcessor as _VocabularyProcessor
except Exception:
    _VocabularyProcessor = object


# Mirroring TensorFLow `VocabularyProcessor`
class VocabularyProcessor(_VocabularyProcessor):
    """ Vocabulary Processor.
    Maps documents to sequences of word ids.
    Arguments:
        max_document_length: Maximum length of documents.
            if documents are longer, they will be trimmed, if shorter - padded.
        min_frequency: Minimum frequency of words in the vocabulary.
        vocabulary: CategoricalVocabulary object.
    Attributes:
        vocabulary_: CategoricalVocabulary object.
    """

    def __init__(self,
                 max_document_length,
                 min_frequency=0,
                 vocabulary=None,
                 tokenizer_fn=None):
        super(VocabularyProcessor, self).__init__(max_document_length,
                                                  min_frequency,
                                                  vocabulary,
                                                  tokenizer_fn)

    def fit(self, raw_documents, unused_y=None):
        """ fit.
        Learn a vocabulary dictionary of all tokens in the raw documents.
        Arguments:
            raw_documents: An iterable which yield either str or unicode.
            unused_y: to match fit format signature of estimators.
        Returns:
            self
        """
        return super(VocabularyProcessor, self).fit(raw_documents, unused_y)

    def fit_transform(self, raw_documents, unused_y=None):
        """ fit_transform.
        Learn the vocabulary dictionary and return indexies of words.
        Arguments:
            raw_documents: An iterable which yield either str or unicode.
            unused_y: to match fit_transform signature of estimators.
        Returns:
            X: iterable, [n_samples, max_document_length] Word-id matrix.
        """
        return super(VocabularyProcessor, self).fit_transform(raw_documents,
                                                              unused_y)

    def transform(self, raw_documents):
        """ transform.
        Transform documents to word-id matrix.
        Convert words to ids with vocabulary fitted with fit or the one
        provided in the constructor.
        Arguments:
            raw_documents: An iterable which yield either str or unicode.
        Yields:
            X: iterable, [n_samples, max_document_length] Word-id matrix.
        """
        return super(VocabularyProcessor, self).transform(raw_documents)

    def reverse(self, documents):
        """ reverse.
        Reverses output of vocabulary mapping to words.
        Arguments:
            documents: iterable, list of class ids.
        Returns:
            Iterator over mapped in words documents.
        """
        return super(VocabularyProcessor, self).reverse(documents)

    def save(self, filename):
        """ save.
        Saves vocabulary processor into given file.
        Arguments:
            filename: Path to output file.
        """
        super(VocabularyProcessor, self).save(filename)

    @classmethod
    def restore(cls, filename):
        """ restore.
        Restores vocabulary processor from given file.
        Arguments:
            filename: Path to file to load from.
        Returns:
            VocabularyProcessor object.
        """
        return super(VocabularyProcessor, cls).restore(filename)