import argparse
from collections import defaultdict

import numpy as np

import config
import models
import utils


def preprocessing(filename: str, num_words: int = 10000, min_count: int = 5):
    """

    :param filename:
    :param num_words:
    :param min_count:
    :return:
    """
    sentences = utils.read_file(filename)
    seqs, tokenizer = utils.tokenize(sentences, num_words, min_count)
    cache = defaultdict(lambda: defaultdict(int))
    utils.build_cooccurrences(sequences=seqs, cache=cache)

    first_indices, second_indices, frequencies = utils.cache_to_pairs(cache=cache)
    return tokenizer, first_indices, second_indices, frequencies


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
    save_model: str = None,
):
    """

    :param first_indices:
    :param second_indices:
    :param frequencies:
    :param epochs:
    :param batch:
    :param vector_size:
    :param vocab_size:
    :param alpha:
    :param lr:
    :param x_max:
    :param save_model:
    :return:
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

    glove.model.save(save_model)
    return glove.model


def main(
    path_data: str,
    epochs: int,
    batch: int,
    vector_size: int,
    save_model: str,
    num_words: int,
    min_count: int,
    alpha: float,
    lr: float,
    x_max: int,
    save_mode: int,
):
    print("Preprocessing...")
    tokenizer, first_indices, second_indices, freq = preprocessing(
        path_data, num_words, min_count
    )

    vocab_size = tokenizer.num_words + 1
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
        save_model=save_model,
    )
    print("Saving vocab...")
    utils.save_vocab(config.VOCAB, tokenizer, vocab_size)
    print("Saving embeddings file...")
    utils.save_word2vec_format(
        model, config.EMBEDDINGS, tokenizer, vector_size, vocab_size, save_mode
    )


def parse_args():
    parser = argparse.ArgumentParser("GloVe train")
    parser.add_argument(help="paths to the corpora", dest="input")
    parser.add_argument(
        "-m",
        help="path where to save the model file",
        dest="model",
        default=config.SAVE_MODEL,
    )
    parser.add_argument(
        "--epochs", help="number of epochs", dest="epochs", type=int, default=5
    )
    parser.add_argument(
        "--batch-size", help="size of the batch", dest="batch", type=int, default=512
    )
    parser.add_argument(
        "--size", help="number of epochs", dest="vector_size", type=int, default=30
    )
    parser.add_argument(
        "--max-vocab",
        help="number of most frequent words to keep",
        dest="num_words",
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
        default=0,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        path_data=args.input,
        epochs=args.epochs,
        batch=args.batch,
        vector_size=args.vector_size,
        save_model=args.model,
        num_words=args.num_words,
        min_count=args.min_count,
        alpha=args.alpha,
        lr=args.lr,
        x_max=args.x_max,
        save_mode=args.save_mode,
    )
