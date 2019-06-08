import argparse
from collections import defaultdict

import numpy as np

import config
import models
from utils import read_file, tokenize, build_cooccurrences, cache_to_pairs


def preprocessing(filename: str, num_words: int = 10000, num_lines: int = 1000):
    """

    :param filename:
    :param num_words:
    :param num_lines:
    :return:
    """
    sentences = read_file(filename, num_lines=num_lines)
    seqs, tokenizer = tokenize(lines=sentences, num_words=num_words)
    cache = defaultdict(lambda: defaultdict(int))
    build_cooccurrences(sequences=seqs, cache=cache)

    first_indices, second_indices, frequencies = cache_to_pairs(cache=cache)
    return tokenizer, first_indices, second_indices, frequencies


def train(
    first_indices: np.array,
    second_indices: np.array,
    frequencies: np.array,
    epochs: int = 5,
    batch: int = 512,
    vector_size: int = 30,
    vocab_size: int = 10000,
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
    :param save_model:
    :return:
    """

    model = models.glove_model(vocab_size + 1, vector_dim=vector_size)
    model.fit(
        [first_indices, second_indices],
        frequencies,
        epochs=epochs,
        batch_size=batch,
        verbose=1,
    )

    model.save(save_model)
    return model


def save_word2vec_format(model, tokenizer, vector_size, vocab_size):
    # saving embeddings
    with open(config.EMBEDDINGS, "w") as f:
        f.write("{} {}\n".format(vocab_size - 1, vector_size))

        vectors = model.get_layer(config.CENTRAL_EMB).get_weights()[0]
        for word, i in tokenizer.word_index.items():
            if i > vocab_size:
                return
            str_vec = " ".join(map(str, list(vectors[i, :])))
            f.write("{} {}\n".format(word, str_vec))


def parse_args():
    parser = argparse.ArgumentParser("GloVe train")
    parser.add_argument(help="paths to the corpora", dest="input")
    parser.add_argument(
        "-o",
        help="path where to save the embeddings file",
        required=True,
        dest="output",
    )
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
        "--vector-size",
        help="number of epochs",
        dest="vector_size",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--num-words",
        help="number of most frequent words to keep",
        dest="num_words",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--num-lines",
        help="number of lines to read. -1 all lines are readed",
        dest="num_lines",
        type=int,
        default=-1,
    )

    return parser.parse_args()


def main(
    filename: str,
    epochs: int,
    batch: int,
    vector_size: int,
    save_model: str,
    num_words: int,
    num_lines: int,
):
    print("Preprocessing...")
    tokenizer, first_indices, second_indices, freq = preprocessing(
        filename, num_words=num_words, num_lines=num_lines
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
        save_model=save_model,
    )
    print("Saving embeddings file...")
    save_word2vec_format(model, tokenizer, vector_size, vocab_size)


if __name__ == "__main__":
    args = parse_args()
    main(
        filename=args.input,
        epochs=args.epochs,
        batch=args.batch,
        vector_size=args.vector_size,
        save_model=args.model,
        num_words=args.num_words,
        num_lines=args.num_lines,
    )
