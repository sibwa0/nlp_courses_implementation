from nltk.tokenize import WordPunctTokenizer 
from collections import Counter
from itertools import chain
import nltk


class BPE():
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def init_vocab(self, corpus: list):
        tokens_vocab = set( "".join(list(chain.from_iterable(corpus))))

        words_counter = Counter( list(chain.from_iterable(corpus)) )

        words_repeat = words_counter.most_common()

        new_tokens_repeat = [tuple( (list(word), count) ) for word, count in words_repeat]
        window_size = 2
        while len(tokens_vocab) < self.vocab_size:
            tokens_counter = Counter()
            tokens_pair = []
            for letters, count in new_tokens_repeat:
                pairs = []
                for ngram in nltk.ngrams(letters, n=window_size):
                    potential_token = "".join( list(ngram) )
                    tokens_counter[potential_token] += count
                    pairs.append(potential_token)
                tokens_pair.append(pairs)

            new_token = tokens_counter.most_common(1)[0][0]
            tokens_vocab.add(new_token)

            for i in range( len(tokens_pair) ):
                if new_token in tokens_pair[i]:
                    for j in range( len(new_tokens_repeat[i][0]) - 1 ):
                        if "".join( new_tokens_repeat[i][0][j : j + window_size] ) == new_token:
                            tokens = new_tokens_repeat[i][0][:j] + [new_token] + new_tokens_repeat[i][0][j + window_size :]

                            tokens_repeat = tuple( (tokens, new_tokens_repeat[i][1]) )

                            new_tokens_repeat[i] = tokens_repeat

                            break

        return tokens_vocab 

if __name__ == "__main__":
    corpus = [["мама", "мыла", "раму", "мама"],
              ["муха", "выла", "лая"]]

    bpe = BPE(vocab_size=15)

    print(bpe.init_vocab(corpus))

    

