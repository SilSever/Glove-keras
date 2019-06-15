import itertools
from collections import defaultdict
from typing import List, Union, Tuple

import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer

from glove_keras import config


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
) -> Tuple[List[List], Tokenizer]:
    """
    Keras Tokenizer to tokenize the lines.
    :param lines: A list of strings.
    :param max_vocab: The maximum number of words to keep.
    :param min_count: Lower limit such that words which occur fewer than <int> times are discarded.
    :return: A list of word indices per sentence, the tokenizer object.
    """
    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(lines)
    print("Num words before:", tokenizer.num_words)
    firs_num_word = itertools.islice(tokenizer.word_counts.items(), max_vocab)
    low_count_words = sum(1 for w, c in firs_num_word if c < min_count)
    tokenizer.num_words -= low_count_words
    print("Num words after:", tokenizer.num_words)

    sequences = tokenizer.texts_to_sequences(lines)
    return sequences, tokenizer


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


def vectors_save_mode(save_mode: int, model):
    """

    :param save_mode:
    :param model:
    :return vectors:
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


def save_word2vec_format(
    model,
    path: str,
    tokenizer: Tokenizer,
    vector_size: int,
    vocab_size: int,
    save_mode: int,
):
    """

    :param model:
    :param path:
    :param tokenizer:
    :param vector_size:
    :param vocab_size:
    :param save_mode:
    :return:
    """
    # saving embeddings
    vectors = vectors_save_mode(save_mode, model)
    with open(path, "w") as f:
        f.write("{} {}\n".format(vocab_size - 1, vector_size))

        for word, i in tokenizer.word_index.items():
            if i > vocab_size:
                return
            str_vec = " ".join(map(str, list(vectors[i, :])))
            f.write("{} {}\n".format(word, str_vec))


def save_vocab(path: str, tokenizer: Tokenizer, vocab_size: int):
    """

    :param path:
    :param tokenizer:
    :param vocab_size:
    :return:
    """
    with open(path, mode="w") as f:
        for word, i in tokenizer.word_index.items():
            if i > vocab_size:
                return
            f.write("{} {}\n".format(word, tokenizer.word_counts[word]))
