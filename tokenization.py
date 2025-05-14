import nltk
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')




# Tokenization pattern: Keep hashtags (#word) and URLs together, split everything else normally
def tokenize_words (text):
    tokens = text

    # Remove punctuation (except in hashtags and URLs, which are preserved by regex)
    tokens = [word for word in tokens if word not in string.punctuation]

    # Remove stopwords
    list_stopwords = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.startswith("#") or word.startswith("http") or word.lower() not in list_stopwords]

    return tokens



def make_numerical_vector (path):

    file = pd.read_csv(path)

    file ["tokens"] = np.where(tokenize_words(file["review"]))
    file ["y-values"] = np.where(file["sentiment"] == 'positive', 1, 0)

    vectorizer = TfidfVectorizer(max_features = 100)
    movies_tfidf_matrix = vectorizer.fit_transform(file["tokens"])

    return
