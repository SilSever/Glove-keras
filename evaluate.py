import argparse
from typing import List

import gensim


def word_similarity(
    words: List[str], path_embeddings: str, top_k: int = 3
) -> List[str]:
    """
    Return the top k similar words for word.
    :param words: words in input.
    :param path_embeddings: embeddings file path.
    :param top_k: number of similar words to retrieve.
    :return: the top k similar words for each word in input
    """
    vectors = gensim.models.KeyedVectors.load_word2vec_format(
        path_embeddings, binary=False
    )
    return [vectors.most_similar(word, topn=top_k) for word in words]


def parse_args():
    parser = argparse.ArgumentParser("GloVe evaluation")
    parser.add_argument(nargs="+", help="words to test", dest="words")
    parser.add_argument(
        "-e", help="paths to the embeddings file", required=True, dest="vectors"
    )
    parser.add_argument(
        "--top", help="number of similar words to retrieve", dest="top", default=3
    )

    return parser.parse_args()


def main(words: List[str], path_embeddings: str, top_k: int = 3):
    word_sim = word_similarity(words, path_embeddings, top_k)
    for i in range(len(words)):
        print("Most similar words to {}:\n{}".format(words[i], word_sim[i]))


if __name__ == "__main__":
    args = parse_args()
    main(args.words, args.vectors, args.top)
