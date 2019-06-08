import codecs

from config import (
    CENTRAL_EMB,
    CONTEXT_EMB,
    CENTRAL_BIASES,
    CONTEXT_BIASES,
    UNKNOWN_WORD,
)
import numpy as np


def write_array_to_file(wf, array):
    for i in range(len(array)):
        wf.write(str(array.item(i)) + " ")
    wf.write("\n")


def save_embeddings(save_filepath, model, vocabulary):

    for layer in model.layers:
        if "_biases" in layer.name or "_embeddings" in layer.name:
            np.save(file=f"{save_filepath}{layer.name}", arr=layer.get_weights()[0])

    weights = np.load(f"{save_filepath}{CENTRAL_EMB}.npy") + np.load(
        f"{save_filepath}{CONTEXT_EMB}.npy"
    )

    vec_path = save_filepath + "vec.txt"
    with codecs.open(vec_path, "w", "utf-8") as wf:
        # First line is vocabulary size and embedding dimension
        wf.write(str(len(vocabulary)) + " " + str(weights.shape[1]) + "\n")
        # Now each line is word "\t" and embedding
        # First word is UNKNOWN_WORD by our convention
        index = 1
        wf.write(UNKNOWN_WORD + "\t")
        write_array_to_file(wf, weights[index])
        index += 1
        # Now emit embedding for each word based on their sorted counts
        sorted_words = reversed(sorted(vocabulary, key=lambda word: vocabulary[word]))
        for word in sorted_words:
            if word == UNKNOWN_WORD:
                continue
            wf.write(word + "\t")
            write_array_to_file(wf, weights[index])
            index += 1
