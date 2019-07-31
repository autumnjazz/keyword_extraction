from gensim.models import Word2Vec
import pandas as pd
from sklearn.manifold import TSNE
from .data_preprocessing import get_data, create_wordlist

def top3_words(text_string):
    
    word_by_sent = create_wordlist(text_string)

    my_model = Word2Vec(sentences=word_by_sent, size=10, window=4, min_count=1, workers=4, sg=0) #min_count 설정 문제
    vectors = my_model.wv
    
    ordered_vocab = [(term, voc.index, voc.count) for term, voc in my_model.wv.vocab.items()]
    ordered_vocab = sorted(ordered_vocab, key=lambda k: -k[2])
    ordered_terms, term_indices, term_counts = zip(*ordered_vocab)
    
    top_words = {}

    for terms in ordered_terms[:3]:
        top_words[terms] = vectors.most_similar(terms)[:4]

    return top_words
