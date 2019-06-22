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


def word_analogy(vectors):
    acc = vectors.accuracy(config.QUESTIONS_WORDS)
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
    return sem_acc, syn_acc, tot_acc


def word_sim(vectors):
    sim_ws353 = vectors.wv.evaluate_word_pairs(config.WS_353)
    sim_rg = vectors.wv.evaluate_word_pairs(config.RG)
    print("Spearman correlations")
    print("WordSim353: {:.3f}".format(sim_ws353[1][0]))
    print("RG: {:.3f}".format(sim_rg[1][0]))
    return sim_ws353, sim_rg


def parse_args():
    parser = argparse.ArgumentParser("GloVe evaluation")
    parser.add_argument(help="paths to the embeddings file", dest="vectors")

    return parser.parse_args()


def main(words: List[str], path_embeddings: str, top_k: int = 3):
    word_sim = word_similarity(words, path_embeddings, top_k)
    for i in range(len(words)):
        print("Most similar words to {}:\n{}".format(words[i], word_sim[i]))


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    args = parse_args()
    print("Loading Gensim embeddings")
    embeddings = KeyedVectors.load_word2vec_format(args.vectors)
    print("Word Analogies")
    word_analogy(embeddings)
    print("Word Similarity")
    word_sim(embeddings)
