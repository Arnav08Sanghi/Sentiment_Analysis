import nltk
import string
from nltk.corpus import stopwords

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


def make_numerical_vector (text):

    



    return tokens
