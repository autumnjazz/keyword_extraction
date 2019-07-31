import re
import string
import nltk
import pandas as pd
from .word_removal import stop_words #stopword 리스트 불러오기
# nltk.download('gutenberg')

# 문제: 단,복수 / 동사 활용

# 텍스트 불러오기 / 기본값 앨리스
def get_data(path=None):
    if path != None:
        document_text = open(path, 'r')
        text_string = document_text.read()
    return text_string


# 문장 나누기
def split_sent(text_string):
    # original_sent = nltk.sent_tokenize(text_string) #구두점 등 포함
    original_sent = re.split("[!?.]+", text_string) #구두점 등 미포함
    return original_sent


# word2vec에 사용할 리스트
def create_wordlist(text_string):
    original_sent = split_sent(text_string)
    word_by_sent = []
    for sent in original_sent:
        sent = re.sub(r"[^a-z]+", " ", sent.lower())
        word_by_sent.append([word for word in nltk.word_tokenize(sent) if word not in stop_words])
    return word_by_sent


# pandas dataframe 생성용
def create_df(text_string):
    original_sent = split_sent(text_string)
    word_for_df = []
    for idx, sent in enumerate(original_sent):
        sent = re.sub(r"[^a-z0-9]+", " ", sent.lower())
        for word,tag in nltk.pos_tag(nltk.word_tokenize(sent)):
            word_for_df.append((idx, word, tag))

    df = pd.DataFrame(data = word_for_df, columns=['sent_num', 'word_for_df', 'pos_tag'])

    # stopwords 분류
    df['is_stopword'] = [True if word in stop_words else False for word in df['word_for_df'] ]

    # 동사, 명사 추출
    # df_nouns = df.loc[(df['pos_tag'].str.startswith('N')),:]
    # df_nouns = df_nouns.loc[(df_nouns['is_stopword'] == False),:]
    # df_verbs = df.loc[(df['pos_tag'].str.startswith('V')),:]
    # df_verbs = df_verbs.loc[(df_verbs['is_stopword'] == False),:]
    return df