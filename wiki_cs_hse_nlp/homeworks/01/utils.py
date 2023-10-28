import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine

import bokeh.models as bm, bokeh.plotting as pl
from pymystem3 import Mystem
import re
from collections import Counter
from itertools import chain
from gensim.models import KeyedVectors
import sys
from unicodedata import category
from string import punctuation


def lemmatize_text(text: str):
    m = Mystem()
    return [m.lemmatize(word) for word in text]


def tokenize(text: str):
    """
    :returns: list of tokenized words
    """

    # codepoints = range(sys.maxunicode + 1)
    # punctuation = {c for i in codepoints if category(c := chr(i)).startswith("P")}
    
    reg = re.compile(r'\w+')
    
    return reg.findall(text.lower())    


def read_corpus(lang: str):
    """
    read corpus of texts with specified language.
    :args:
        lang (string): ru or be
    :returns:
        list of lists, with tokenized words from the corpus
    """

    assert lang in ['ru', 'be', "ru_copy", "be_copy"]

    m = Mystem()
    texts = pd.read_csv(f'{lang}.csv')['text']
    lem_tok_texts = texts.apply(lambda s: tokenize("".join(m.lemmatize(s))))
    
    return lem_tok_texts.to_list()



def get_distinct_words(corpus, min_count=10):
    """ 
    collect a list of distinct words for the corpus.
    :args:
        corpus (list of list of strings): corpus of texts
        min_count (int): ignores all words with total frequency lower than this
    :returns:
        words (list of strings): sorted list of distinct words across the corpus
        word_counter (collections.Counter()): dict that contains for every word in "words" an anount of times the word appears
    """
    words = []
    word_counter = Counter(list(chain.from_iterable(corpus)))

    index = bin_count_counter_index(word_counter.most_common(), min_count)

    word_counter = word_counter.most_common(index)
    words = [elem[0] for elem in word_counter]
    new_word_counter = Counter({elem[0]: elem[1] for elem in word_counter})

    return words, new_word_counter


def plot_embeddings(reduced_matrix, token=None, radius=10, alpha=0.25, show=True, color='blue'):
    """ 
    :args:
        reduced_matrix (np.ndarray [n_words, 2]): matrix of 2-dimensioal word embeddings
        token (list): list of tokens that contains captions for each embedding
    """

    if isinstance(color, str):
        color = [color] * len(reduced_matrix)
    data_source = bm.ColumnDataSource({'x': reduced_matrix[:, 0], 'y': reduced_matrix[:, 1], 'color': color, 'token': token})

    fig = pl.figure(active_scroll='wheel_zoom', width=600, height=400)
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[('token', "@token")]))
    if show:
        pl.show(fig)
    return fig


def eval_simlex(model: KeyedVectors, filename="simlex911"):
    """
    calculates Spearman's correlation between humans' and model's similarity predictions
    """
    simlex = pd.read_csv(f'{filename}.csv')
    sims = []

    for row in simlex.iterrows():

        embed1 = model.get_vector(row[1]['word1'])
        embed2 = model.get_vector(row[1]['word2'])

        sims.append(1 - cosine(embed1, embed2))

    corr = spearmanr(np.array(sims), simlex['similarity'])
    return corr.correlation


def bin_count_counter_index(words_lst: list, min_count: int):
    """
    return index for Counter to get words, which occures min_count times 
    """
    start = 0
    end = len(words_lst) - 1
    cur = -1
    while cur != min_count:
        index = start + (end - start) // 2
        cur = words_lst[index][1]
        if cur < min_count:
            end = index
        elif cur > min_count:
            start = index + 1
    
    next_word_count = words_lst[index + 1][1]
    while next_word_count == min_count:
        index += 1
        if index == end:
            break
        next_word_count = words_lst[index + 1][1]

    return index + 1