import argparse
from collections import defaultdict

import config
from model import glove_model
from utils import read_file, tokenize, build_cooccurrences, cache_to_pairs


def preprocessing(filename: str):
    """

    :param filename:
    :return:
    """
    sentences = read_file(filename)
    seqs, tokenizer = tokenize(lines=sentences)

    cache = defaultdict(lambda: defaultdict(int))
    build_cooccurrences(sequences=seqs, cache=cache)

    first_indices, second_indices, frequencies = cache_to_pairs(cache=cache)
    return tokenizer, first_indices, second_indices, frequencies


def train(
    filename: str,
    epochs: int = 5,
    batch: int = 512,
    vector_size: int = 30,
    save_model: str = None,
):
    """

    :param filename:
    :param epochs:
    :param batch:
    :param vector_size:
    :param save_model:
    :return:
    """
    print("Preprocessing...")
    token, first_indices, second_indices, freq = preprocessing(filename)

    print("Training...")
    model = glove_model(token.num_words + 1, vector_dim=vector_size)
    model.fit(
        [first_indices, second_indices],
        freq,
        epochs=epochs,
        batch_size=batch,
        verbose=1,
    )

    model.save(save_model)


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
    parser.add_argument("--epochs", help="number of epochs", dest="epochs", default=5)
    parser.add_argument(
        "--batch_size", help="number of epochs", dest="batch", default=512
    )
    parser.add_argument(
        "--vector-size", help="number of epochs", dest="vector_size", default=30
    )
    parser.add_argument(
        "--min-count",
        help="ignores all words with total frequency lower than this",
        dest="min_count",
        default=3,
    )

    return parser.parse_args()


def main(filename: str, epochs: int, batch: int, vector_size: int, save_model: str):
    train(
        filename,
        epochs=epochs,
        batch=batch,
        vector_size=vector_size,
        save_model=save_model,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        filename=args.input,
        epochs=args.epochs,
        batch=args.batch,
        vector_size=args.vector_size,
        save_model=args.model,
    )
