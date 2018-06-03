import os
import sys
import math
import matplotlib.pyplot as plt
import nltk
import scipy

import pymorphy2

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import word_tokenize,sent_tokenize

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
from wordcloud import WordCloud

import pymongo

nltk.download('punkt')
nltk.download('stopwords')

mongo_url = os.getenv('PYTHON_LAB3_MONGO_URL', None)
if (mongo_url == None):
    print("Can't connect mongo")
    sys.exit(0)

connection = pymongo.MongoClient(mongo_url)
db = connection["python_lab2"]

NUMBER_OF_CLUSTERS = 10
current_dir = os.path.dirname(__file__)
morph = pymorphy2.MorphAnalyzer()


def normalize_word(word):
    w = morph.parse(word)[0]
    return w.normal_form


def prepare_comments(messages):
    prepared = []
    for message in messages:
        words = word_tokenize(str.lower(message))
        normalized = [normalize_word(word) for word in words
            if word.isalpha() and word not in stopwords.words("russian")]
        if len(normalized) > 0:
            prepared.append(normalized)

    for sentences in prepared:
        toReturn = [' '.join(word for word in sentences) for sentences in prepared]

    return toReturn


def vectorize(messages):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(messages)


def clusterize(number_of_clusters, matrix):
    model = KMeans(n_clusters=number_of_clusters, random_state=0, n_jobs=-2)
    model.fit(matrix)
    return model.predict(matrix)


def build_clouds(number_of_clusters, model, messages, filepath, filename):
    for cluster_num in range(number_of_clusters):
        text = ""
        for i, cluster in enumerate(model):
            if cluster_num == cluster:
                text += messages[i] + " "
        if len(text) > 0:
            wc = WordCloud(background_color='white',
                                    max_words=200,
                                    stopwords=stopwords.words("russian"))
            wc.generate(text)
            full_filename = '{}/{}{}.png'.format(filepath, filename, cluster_num+1)
            wc.to_file(os.path.join(current_dir, full_filename))
            

def get_text_from_comments():
    comments = db.comments.distinct("text")
    return comments
    

if __name__ == "__main__":
    comments = get_text_from_comments()
    prepared_messages = prepare_comments(comments)
    matrix = vectorize(prepared_messages)
    model = clusterize(NUMBER_OF_CLUSTERS, matrix)

    build_clouds(NUMBER_OF_CLUSTERS, model, prepared_messages, 'tags', 'tag')