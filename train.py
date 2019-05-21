from collections import defaultdict
from utils import read_file, tokenize, build_cooccurrences, cache_to_pairs

from model import glove_model

from config import IN_FILE, SAVE_MODEL


def preprocessing():

    sentences = read_file(IN_FILE)

    seqs, tokenizer = tokenize(lines=sentences)

    cache = defaultdict(lambda: defaultdict(int))
    build_cooccurrences(sequences=seqs, cache=cache)

    first_indices, second_indices, frequencies = cache_to_pairs(cache=cache)
    return tokenizer, first_indices, second_indices, frequencies


def train(epochs=5, batch=512, vector_size=30):

    print('Preprocessing...')
    token, first_indices, second_indices, freq = preprocessing()

    print('Training...')
    model = glove_model(token.num_words + 1, vector_dim=vector_size)
    model.fit([first_indices, second_indices],
              freq,
              epochs=epochs,
              batch_size=batch,
              verbose=1)

    model.save(SAVE_MODEL)


def main():
    train()


if __name__ == '__main__':
    main()
