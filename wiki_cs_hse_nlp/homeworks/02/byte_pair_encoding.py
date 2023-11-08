import enum
from nltk.tokenize import WordPunctTokenizer 
from collections import Counter
from itertools import chain
import nltk


class my_BPE():
    def __init__(self,
                 vocab_size=30_000,
                 special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens


    def train(self, corpus: list):
        tokens_vocab = set( "".join(list(chain.from_iterable(corpus))) )

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
        
        tokens_vocab = tokens_vocab.union( self.special_tokens )
        self.token_to_ind = { token: ind for ind, token in enumerate(tokens_vocab) }
        self.ind_to_token = { ind: token for ind, token in enumerate(tokens_vocab) }


    def encode(self, string: str):
        ids = []

        for word in string.split():
        
            start = 0
            while start != len(word):
                
                end = len(word)
                while word[start : end] not in self.token_to_ind and end > start:
                    end -= 1
                
                if start == end:
                    ids.append( self.token_to_ind["[UNK]"] )
                    start += 1
                else:
                    ids.append( self.token_to_ind[word[start : end]] )
                    start = end

        return ids
    

    def decode(self, ids: list):
        tokens = [self.ind_to_token[id] for id in ids]

        return tokens


if __name__ == "__main__":
    corpus = [["мама", "мыла", "раму", "мама"],
              ["муха", "выла", "лая"]]

    bpe = my_BPE(vocab_size=15)

    bpe.train(corpus)

    ids = bpe.encode("мама мыла )рамув") 

    print( ids )

    print( bpe.decode(ids) )

    

