import nltk
import string
import pandas as pd
import numpy as np
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from joblib import dump

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')




# Tokenization pattern: Keep hashtags (#word) and URLs together, split everything else normally
def tokenize_words (text):
    tokens = word_tokenize(text)

    # Remove punctuation (except in hashtags and URLs, which are preserved by regex)
    tokens = [word for word in tokens if word not in string.punctuation]

    # Remove stopwords
    list_stopwords = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in list_stopwords and not word.startswith('#') and not word.startswith('http')]


    return tokens




def make_numerical_vector (path):

    file = pd.read_csv(path)

    file["tokens"] = file["review"].apply(tokenize_words)
    print(file["tokens"])
    y_value = np.where(file["sentiment"] == 'positive', 1, 0)
    print(y_value)
    file["tokens_joined"] = [" ".join(tokens) for tokens in file["tokens"]]
    vectorizer = TfidfVectorizer(
                    max_features = 10000,
                    ngram_range=(1,2), #Uniword and Biword 
                    min_df=5,
                    max_df=0.95)
    movies_tfidf_matrix = vectorizer.fit_transform(file["tokens_joined"])
    print(movies_tfidf_matrix)
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    x_train, x_test, y_train, y_test = train_test_split(movies_tfidf_matrix, y_value, test_size=0.2, random_state = 18)

    return x_train, x_test, y_train, y_test




