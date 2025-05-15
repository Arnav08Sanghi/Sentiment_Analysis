import nltk
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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
    #file ["y-values"] = np.where(file["sentiment"] == 'positive', 1, 0)

    vectorizer = TfidfVectorizer(max_features = 100)
    movies_tfidf_matrix = vectorizer.fit_transform(file["tokens"])
    y_value = np.where(file["sentiment"] == 'positive', 1, 0)


    x_train, x_test, y_train, y_test = train_test_split(movies_tfidf_matrix, y_value, test_size=0.2, random_state = 18)


    return x_train, x_test, y_train, y_test
