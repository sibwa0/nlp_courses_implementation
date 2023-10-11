import numpy as np
from utils import (
    get_distinct_words,
    tokenize   
)


def compute_co_occurrence_matrix(corpus, window_size=5):
    """
    compute co-occurrence matrix for the given corpus and window_size.

    :args:
        corpus (list of list of strings): corpus of texts
        window_size (int): size of context window
    :return:
        matrix (a symmetric numpy matrix [vocab_size, vocab_size]): co-occurence matrix of word counts. 
        The ordering of the words in the rows/columns should be the same as the ordering of the words given by the get_distinct_words function.
    """
    def pad_text(text: list, window_size: int, pad: str):
        appendix = [pad] * window_size

        return appendix + text + appendix

    words, _ = get_distinct_words(corpus, min_count=2)
    token_to_ind = {word: i for i, word in enumerate(words)}
    matrix = np.zeros((len(words), len(words)))

    for text in corpus:
        text = pad_text(text, window_size=window_size, pad="UNK")
        for ind in range(len(text) - 2 * window_size): 
            start = ind
            end = start + 2 * window_size
            w_j = start + window_size
            if text[w_j] in token_to_ind.keys():
                for c in range(start, end + 1):
                    if c != w_j and text[c] in token_to_ind.keys():
                        matrix[token_to_ind[text[w_j]], token_to_ind[text[c]]] += 1
    
    
    return matrix

bart_texts = [
    'Я больше не буду тратить мел.',
    'Я не буду кататься на скейте по зданию школы.',
    'Я не буду рыгать в классе.',
    'Я не буду провоцировать революцию в школе.',
    'Я не буду рисовать голых девушек в классе.',
    'Я не видел Элвиса.',
    'Я не буду называть свю учительницу "Горячие пирожные".',
    'Жвачка с чесноком - это не смешно.',
    'Они смеются надо мной, а не со мной.',
    'Я не буду кричать "ПОЖАР!" в заполненном классе.'
]

co_occurence_matrix = np.array([
       [0., 8., 7., 0., 0., 2.],
       [8., 0., 6., 0., 0., 0.],
       [7., 6., 0., 1., 0., 0.],
       [0., 0., 1., 0., 3., 0.],
       [0., 0., 0., 3., 0., 0.],
       [2., 0., 0., 0., 0., 0.]
])



if __name__ == "__main__":
    bart_texts = [tokenize(text) for text in bart_texts]

    co_oc = compute_co_occurrence_matrix(bart_texts, window_size=2)
    print(co_oc)

    assert np.allclose(co_oc, co_occurence_matrix)