from nltk.tokenize import WordPunctTokenizer




def texts_corpus(texts: list):
    """
    :param texts: list of strings
    :return corpus: list of tokens list
    """
    wpt = WordPunctTokenizer()

    corpus = [wpt.tokenize(text.lower()) for text in texts]

    return corpus
