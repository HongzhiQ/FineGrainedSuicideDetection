import pandas as pd
import jieba
import synonyms
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import random


def extract_topk_keywords(texts, k=1):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names_out()
    top_k_words = set([features[i] for i in indices[:k]])
    return top_k_words


def replace_with_synonyms(sentence, top_k_words, num_replacements=4):
    words = list(jieba.cut(sentence))
    new_sentences = []
    skip_words = {"的", "了", "着","啊","吧","谁","吗"}  # Add a vocabulary set without replacement

    for _ in range(num_replacements):
        new_words = words.copy()
        for i, word in enumerate(words):
            if word not in top_k_words and word not in skip_words and random.random() < 0.7:  # skip meaningless words
                synonyms_list = synonyms.nearby(word)[0][:5]
                if synonyms_list:
                    valid_synonyms = [syn for syn in synonyms_list if len(syn) == len(word)]
                    if valid_synonyms:
                        new_word = random.choice(valid_synonyms)
                    else:
                        new_word = random.choice(synonyms_list)
                    new_words[i] = new_word
        new_sentences.append(''.join(new_words))

    return new_sentences


df = pd.read_csv('../../data/fine-grained/fold5.tsv', sep='\t')
#selected_df = df[df['myLabel'].isin([0,1,2,3,5,6,7,8,9,10])]
selected_df = df[df['myLabel'].isin([10])]
texts = selected_df['comment'].tolist()
# Extract Top-K keywords
top_k_words = extract_topk_keywords(texts)
new_df = pd.DataFrame(columns=['comment', 'myLabel'])
# Process each sentence
for index, row in selected_df.iterrows():
    new_commands = replace_with_synonyms(row['comment'], top_k_words)
    for new_command in new_commands:
        new_df = new_df._append({'comment': new_command, 'myLabel': row['myLabel']}, ignore_index=True)
# Write new data to file
new_df.to_csv('', sep='\t', index=False)