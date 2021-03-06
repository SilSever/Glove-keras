import argparse
import os

import numpy as np

import config
import models
import utils


def preprocessing(
    filename: str, max_vocab: int = 10000, min_count: int = 5, window: int = 15
):
    """
    Preprocess the input file, building the co-occurence matrix.
    :param filename: Path of the input file.
    :param max_vocab: Number of most frequent words to keep.
    :param min_count: Lower limit such that words which occur fewer than <int> times are discarded.
    :param window: Number of context words to the left and to the right.
    :return: The co-occurence matrix unpacked.
    """
    sentences = utils.read_file(filename)
    seqs, word_index, word_counts = utils.tokenize(sentences, max_vocab, min_count)
    print("Building cooccurences matrix...")
    cooccurence_dict = utils.build_cooccurrences(sentences=seqs, window=window)
    first_indices, second_indices, frequencies = utils.unpack_cooccurrence(
        cooccurence_dict=cooccurence_dict
    )
    return first_indices, second_indices, frequencies, word_index, word_counts


def train(
    first_indices: np.array,
    second_indices: np.array,
    frequencies: np.array,
    epochs: int = 5,
    batch: int = 512,
    vector_size: int = 30,
    vocab_size: int = 10000,
    alpha: float = 0.75,
    lr: float = 0.05,
    x_max: int = 100,
):
    """
    Train the Keras GloVe model.
    :param first_indices: np.array of the first indices of the co-occurence matrix.
    :param second_indices: np.array of the second indices of the co-occurence matrix.
    :param frequencies:
    :param epochs: Number of epochs.
    :param batch: Size of the batch.
    :param vector_size: Size of the vectors.
    :param vocab_size: Size of the vocabulary.
    :param alpha: Parameter in exponent of weighting function.
    :param lr: Parameter specifying cutoff in weighting function.
    :param x_max: Parameter specifying cutoff in weighting function.
    :param path_model: Path where to save the model.
    :return: The model.
    """

    glove = models.Glove(
        vocab_size + 1, vector_dim=vector_size, alpha=alpha, lr=lr, x_max=x_max
    )

    glove.model.fit(
        [first_indices, second_indices],
        frequencies,
        epochs=epochs,
        batch_size=batch,
        verbose=1,
    )

    glove.model.save(config.SAVE_WEIGHTS)
    return glove.model


def main(
    path_data: str,
    epochs: int,
    batch: int,
    vector_size: int,
    window: int,
    path_vectors: str,
    max_vocab: int,
    min_count: int,
    alpha: float,
    lr: float,
    x_max: int,
    save_mode: int,
):
    print("Preprocessing...")
    first_indices, second_indices, freq, word_index, word_counts = preprocessing(
        path_data, max_vocab, min_count, window
    )

    vocab_size = len(word_counts) + 1
    print("Vocab size:", vocab_size)
    print("Training...")
    model = train(
        first_indices=first_indices,
        second_indices=second_indices,
        frequencies=freq,
        epochs=epochs,
        batch=batch,
        vector_size=vector_size,
        vocab_size=vocab_size,
        alpha=alpha,
        lr=lr,
        x_max=x_max,
    )
    print("Saving vocab...")
    utils.save_vocab(config.VOCAB, word_counts)
    print("Saving embeddings file...")
    # path_folder = config.EMBEDDINGS.split("/")[0]
    # if not os.path.isdir(path_folder):
    #     os.mkdir(path_folder)
    utils.save_word2vec_format(
        model, path_vectors, word_index, vector_size, save_mode
    )


def parse_args():
    parser = argparse.ArgumentParser("GloVe train")
    parser.add_argument(help="paths to the corpora", dest="input")
    parser.add_argument(
        "-o",
        help="path where to save the vectors",
        dest="vectors",
        default=config.EMBEDDINGS,
    )
    parser.add_argument(
        "--epochs", help="number of epochs", dest="epochs", type=int, default=5
    )
    parser.add_argument(
        "--batch-size", help="size of the batch", dest="batch", type=int, default=512
    )
    parser.add_argument(
        "--size", help="size of the vectors", dest="vector_size", type=int, default=30
    )
    parser.add_argument(
        "--window",
        help="number of context words to the left and to the right",
        dest="window",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--max-vocab",
        help="number of most frequent words to keep",
        dest="max_vocab",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--min-count",
        help="lower limit such that words which occur fewer than <int> times are discarded",
        dest="min_count",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--alpha",
        help="parameter in exponent of weighting function",
        dest="alpha",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--lr", help="initial learning rate", dest="lr", type=float, default=0.05
    )
    parser.add_argument(
        "--x-max",
        help="parameter specifying cutoff in weighting function",
        dest="x_max",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--save-mode",
        help="save mode determines the type of embeddings to save",
        dest="save_mode",
        type=int,
        default=2,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        path_data=args.input,
        epochs=args.epochs,
        batch=args.batch,
        vector_size=args.vector_size,
        window=args.window,
        path_vectors=args.vectors,
        max_vocab=args.max_vocab,
        min_count=args.min_count,
        alpha=args.alpha,
        lr=args.lr,
        x_max=args.x_max,
        save_mode=args.save_mode,
    )
