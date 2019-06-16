import argparse
import logging
from typing import List

from gensim.models import KeyedVectors

import config


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
    vectors = KeyedVectors.load_word2vec_format(path_embeddings, binary=False)
    return [vectors.most_similar(word, topn=top_k) for word in words]


def fancy_w2v_operation(path_embeddings):
    vectors = KeyedVectors.load_word2vec_format(path_embeddings, binary=False)
    print(vectors.most_similar(positive=["woman", "king"], negative=["man"], topn=2))


def evaluation(path_embeddings: str):
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    accuracies = []
    print("Loading Gensim embeddings")
    vectors = KeyedVectors.load_word2vec_format(path_embeddings)
    accuracies.append(print_accuracy(vectors, config.QUESTIONS_WORDS))


def print_accuracy(model, questions_file):
    print("Evaluating...")
    acc = model.accuracy(questions_file)

    sem_correct = sum((len(acc[i]["correct"]) for i in range(5)))
    sem_total = sum(
        (len(acc[i]["correct"]) + len(acc[i]["incorrect"])) for i in range(5)
    )
    sem_acc = 100 * float(sem_correct) / sem_total
    print("Semantic: {}/{}, Accuracy: {:.2f}%".format(sem_correct, sem_total, sem_acc))

    syn_correct = sum((len(acc[i]["correct"]) for i in range(5, len(acc) - 1)))
    syn_total = sum(
        (len(acc[i]["correct"]) + len(acc[i]["incorrect"]))
        for i in range(5, len(acc) - 1)
    )
    syn_acc = 100 * float(syn_correct) / syn_total
    print("Syntactic: {}/{}, Accuracy: {:.2f}%".format(syn_correct, syn_total, syn_acc))
    tot_correct = sem_correct + syn_correct
    tot_total = sem_total + syn_total
    tot_acc = 100 * float(tot_correct) / tot_total
    print("Total: {}/{}, Accuracy: {:.2f}%".format(tot_correct, tot_total, tot_acc))
    return sem_acc, syn_acc


def parse_args():
    parser = argparse.ArgumentParser("GloVe evaluation")
    parser.add_argument(help="paths to the embeddings file", dest="vectors")

    return parser.parse_args()


def main(words: List[str], path_embeddings: str, top_k: int = 3):
    word_sim = word_similarity(words, path_embeddings, top_k)
    for i in range(len(words)):
        print("Most similar words to {}:\n{}".format(words[i], word_sim[i]))


if __name__ == "__main__":
    args = parse_args()
    evaluation(args.vectors)
