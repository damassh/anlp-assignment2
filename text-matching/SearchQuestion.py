import csv
import nltk as nltk
import pandas as pd
import numpy as np
import sys
import re
from string import digits
from string import punctuation
from collections import defaultdict

import warnings

warnings.filterwarnings("ignore")
remove_digits = str.maketrans('', '', digits)
punct = (punctuation + 'ÔøΩüčé')

if len(sys.argv) < 2:
    print('Format: Python SearchQuestion.py "your question" ')
    sys.exit(1)

query = sys.argv[1]

# Load the dataset
dataset = pd.read_csv('data.tsv', sep='\t', error_bad_lines=False, quoting=csv.QUOTE_NONE)
dataset.dropna(inplace=True)


def preprocessing(doc):
    # remove punctuation and special characters
    translation = dict.fromkeys(map(ord, punct), None)  # Dictionary with punctuation to be removed
    doc = doc.lower().translate(translation)

    # remove non-english words
    doc = re.sub('([^\x00-\x7F])+', '', doc)

    # remove numbers
    doc = re.sub('[0-9]+', '', doc)

    return nltk.word_tokenize(doc)


# Apply text preprocessing techniques for question 1 and question 2 documents
dataset['q2_tokens'] = dataset['question2'].apply(preprocessing)
question2 = dataset['question2'].tolist()
q2_tokens = dataset['q2_tokens'].tolist()
q1_tokens = [preprocessing(query)]
dim = 300


# Text matching with sentence embedding by averaging word embedding
# load GloVe data model
def load_glove():
    filename = 'glove.6B/glove.6B.300d.txt'
    glove_data = defaultdict()
    file = open(filename, "r")

    for line in file:
        reader = line.split()
        word = reader[0]
        glove_data[word] = np.array([float(val) for val in reader[1:]])
    return glove_data


def get_sentence_embedding(docs, dim):
    sentence_embedding = np.zeros([len(docs), dim])
    glove_corpus = set(glove_data.keys())

    for index, doc in enumerate(docs):
        word_embedding = np.zeros([1, dim])
        word_embedding_count = 0
        average = 0
        for word in doc:
            # sum the word weight retrieved from glove and add to word embedding
            if word in glove_corpus:
                word_embedding += glove_data[word]
                word_embedding_count += 1

        # calculate average
        if word_embedding_count > 0:
            average = word_embedding / word_embedding_count
        else:
            average = word_embedding_count

        sentence_embedding[index, :] = average
    return sentence_embedding


def sentence_text_matching(q1_s_embedding, q2_s_embedding, rank_num):
    cos_sim = np.dot(q2_s_embedding, q1_s_embedding.reshape([-1, 1])) / np.linalg.norm(q1_s_embedding)
    cos_sim = cos_sim.flatten()
    cos_sim = cos_sim / np.linalg.norm(q2_s_embedding, axis=1)
    cos_sim[np.isnan(cos_sim)] = -1

    '''
    Document ranking
    retrieve top 50 matches
    '''
    idx = np.argpartition(cos_sim, -50)[-50:]
    top_doc = idx[np.argsort(cos_sim[idx])[::-1]]

    # remove duplicates and select the question based on top ranks
    text_match = []
    for doc_id in top_doc:
        question = question2[doc_id]
        if question not in text_match:
            text_match += [question]
            if len(text_match) == rank_num: break
    return text_match


# load glove dataset
glove_data = load_glove()

# retrieve sentence embedding for q1 and q2
q1_sentence_embedding = get_sentence_embedding(q1_tokens, dim)
q2_sentence_embedding = get_sentence_embedding(q2_tokens, dim)

# retrieve top 5 results
text_match = []
for query in q1_sentence_embedding:
    text_match += [sentence_text_matching(query, q2_sentence_embedding, 5)]

# print the top 5 results
for match in text_match:
    counter = 1
    for m in match:
        print(counter, '.', m)
        counter+=1