import itertools
import multiprocessing
from collections import defaultdict, OrderedDict, Counter
from functools import partial
from multiprocessing import Pool
from typing import List, Union, Tuple, Dict

import nltk
import numpy as np

import config


def read_file(filename: str) -> List[str]:
    """
    Read the dataset line by line.
    :param filename: file to read
    :return: a list of lines
    """
    with open(filename, mode="r", encoding="utf8") as file:
        f = (line.strip() for line in file)
        return [line for line in f if line]


def tokenize(
    lines: List[str], max_vocab: int = 10000, min_count: int = 5
) -> Tuple[List[List], Dict[str, int], Dict[str, int]]:
    """
    Tokenize the lines.
    :param lines: A list of strings.
    :param max_vocab: The maximum number of words to keep.
    :param min_count: Lower limit such that words which occur fewer than <int> times are discarded.
    :return: A list of word indices per sentence, the word frequences dict and the word index dict.
    """
    print("Tokenization of the file...")
    lines_tokenized = [nltk.word_tokenize(l) for l in lines]
    print("Building word vocabs...")
    word_counts = build_word_counts(lines_tokenized)
    max_vocab_counts = itertools.islice(word_counts.items(), max_vocab)
    word_counts_reduced = {w: c for w, c in max_vocab_counts if c >= min_count}
    word_index = build_word_index(word_counts_reduced)
    print("Number of tokens:", len(word_index))
    print("Texts to sequences...")
    sequences = texts_to_sequences(lines_tokenized, word_index)
    return sequences, word_index, word_counts_reduced


def build_word_counts(lines: List[List[str]]) -> OrderedDict:
    """
    Builds a dictionary of word frequences.
    :param lines: A list of list of strings.
    :return: an ordered dict of word frequences.
    """
    word_counts = Counter(w for l in lines for w in l)
    return OrderedDict(
        sorted(word_counts.items(), key=lambda k: int(k[1]), reverse=True)
    )


def build_word_index(word_counts: Dict) -> Dict:
    """
    Builds a dictionary from word to index.
    :param word_counts: Dictionary of word frequences.
    :return: a dicitonary word -> index.
    """
    return {k: i for i, k in enumerate(word_counts)}


def word_to_index(word_index, word):
    """
    Partial function of texts_to_sequences used for multiprocessing
    :param word_index: dictionary of word indexes.
    :param word: just a string
    :return: a list with just an element
    """
    if word in word_index:
        return word_index[word]


def texts_to_sequences(lines: List[List[str]], word_index: Dict[str, int]):
    """
    Transforms each text to a sequence of integers.
    :param lines: A list of list of strings.
    :param word_index: dictionary of word indexes.
    :return: A list of sequences.
    """
    return [
        list(
            filter(
                None,
                Pool(multiprocessing.cpu_count()).map(
                    partial(word_to_index, word_index), line
                ),
            )
        )
        for line in lines
    ]


def build_cooccurrences(sentences: List[List[int]], window: int = 15):
    """
    It updates a shared cache for by iteratively calling 'bigram_count'
    :param sentences: The input file.
    :param window: Number of context words to the left and to the right.
    :return: A dictionary containing the distane between each word.
    """
    cooccurence_dict = defaultdict(lambda: defaultdict(int))
    for s in sentences:
        distance_count(
            token_list=s, window_size=window, cooccurence_dict=cooccurence_dict
        )
    return cooccurence_dict


def distance_count(
    token_list: List[int], window_size: int, cooccurence_dict: defaultdict
):
    """
    It computes the co-occurrence matrix required by GloVe as a dictionary.
    :param token_list: The representation of a sentence as a list of integers (word indices)
    :param window_size: The size of the window around the central word
    :param cooccurence_dict: A dictionary containing the distane between each word.
    """
    sentence_size = len(token_list)

    for central_index, central_word_id in enumerate(token_list):
        for distance in range(1, window_size + 1):
            if central_index + distance < sentence_size:
                first_id, second_id = sorted(
                    [central_word_id, token_list[central_index + distance]]
                )
                cooccurence_dict[first_id][second_id] += 1.0 / distance


def unpack_cooccurrence(
    cooccurence_dict: defaultdict
) -> Union[np.array, np.array, np.array]:
    """
    :param cooccurence_dict: A dictionary containing the distane between each word.
    :return: the cooccurence dictionary unpacked.
    """
    first, second, x_ijs = [], [], []
    for first_id in cooccurence_dict.keys():
        for second_id in cooccurence_dict[first_id].keys():
            x_ij = cooccurence_dict[first_id][second_id]
            # add (main, context) pair
            first.append(first_id)
            second.append(second_id)
            x_ijs.append(x_ij)
            # add (context, main) pair
            first.append(second_id)
            second.append(first_id)
            x_ijs.append(x_ij)

    return np.array(first), np.array(second), np.array(x_ijs)


def save_word2vec_format(
    model, path: str, word_index: Dict, vector_size: int, save_mode: int
):
    """
    Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.
    :param model: keras model.
    :param path: path where to save the vectors.
    :param word_index: dictionary of word idexes.
    :param vector_size: size of the vectors
    :param save_mode: modes for word vector output
        0: output all data, for both word and context word vectors, including bias terms
        1: output word vectors, excluding bias terms
        2: output word vectors + context word vectors, excluding bias terms
    :return:
    """
    # saving embeddings
    vectors = _vectors_save_mode(save_mode, model)
    with open(path, "w") as f:
        f.write("{} {}\n".format(len(word_index) - 1, vector_size))

        for word, i in word_index.items():
            str_vec = " ".join(map(str, list(vectors[i, :])))
            f.write("{} {}\n".format(word, str_vec))


def _vectors_save_mode(save_mode: int, model):
    """
    Modes for word vector output.
    :param save_mode:
        0: output all data, for both word and context word vectors, including bias terms
        1: output word vectors, excluding bias terms
        2: output word vectors + context word vectors, excluding bias terms
    :param model: model
    :return vectors: vectors
    """

    if save_mode is 0:
        # Save (word + context word) vectors (with biases)
        print("Mode is", save_mode)
        word_vectors = (
            model.get_layer(config.CNTRL_EMB).get_weights()[0]
            + model.get_layer(config.CNTRL_BS).get_weights()[0]
        )

        context_vectors = (
            model.get_layer(config.CTX_EMB).get_weights()[0]
            + model.get_layer(config.CTX_BS).get_weights()[0]
        )

        vectors = word_vectors + context_vectors

    elif save_mode is 1:
        # save word vectors (no bias)
        print("Mode is", save_mode)
        vectors = model.get_layer(config.CNTRL_EMB).get_weights()[0]

    else:
        # Save (word + context word) vectors (no biases)
        print("Mode is", save_mode)
        vectors = (
            model.get_layer(config.CNTRL_EMB).get_weights()[0]
            + model.get_layer(config.CTX_EMB).get_weights()[0]
        )
    return vectors


def save_vocab(path: str, word_counts: Dict):
    """
    Save word with frequences as file.
    :param path: path where to save the vocab.
    :param word_counts: dictionary with words and frequences.
    :return:
    """
    with open(path, mode="w") as f:
        for word, i in word_counts.items():
            f.write("{} {}\n".format(word, word_counts[word]))
