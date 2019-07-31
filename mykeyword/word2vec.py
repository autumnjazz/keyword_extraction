from gensim.models import Word2Vec
import pandas as pd
from sklearn.manifold import TSNE
from .data_preprocessing import get_data, create_wordlist

def top3_words(text_string):
    if text_string == 'test':
        text_string = """
            Story highlights Don't be fooled by the word "energy"

            Some energy bars contain as much saturated fat as a Snickers bar

            Energy bars are a convenient source of nutrition and come in a wide variety of flavors to satisfy different palates. They are often fortified with vitamins and minerals, which can help fill nutritional gaps.

            But, like many foods in a specific category, not all energy bars are created equal. Those that are low in saturated fat and sugars, with a decent amount of protein and fiber, can provide a nutritious, satisfying pick-me-up. Others can closely mimic a candy bar. For example, some bars covered in chocolate contain as much saturated fat as a Snickers bar; others contain almost as much sugar.

            Granola bars are a convenient source of nutrition, but can vary significantly in terms of nutrition.

            Energy bars containing mostly fruit and nuts can serve as satisfying snacks. But if you're looking for a meal replacement, aim for a bar with a higher amount of protein: about 10 to 20 grams. Athletes can also benefit from choosing a bar with more protein and carbohydrates, as their needs are higher.

            You can afford more calories if bars are consumed in place of meals and not as snacks. But if a bar is intended only to tide you over until dinner, limit it to 150 to 200 calories.

            In general, try to aim for bars with less than 3 grams of saturated fat and at least 4 grams of fiber. Palm kernel oil in yogurt and chocolate coatings will boost saturated fat. Also watch out for bars with ingredients such as brown rice syrup or cane invert syrup listed first, as they are generally higher in sugars than others and are better suited for athletes, not weight watchers.

        """
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
