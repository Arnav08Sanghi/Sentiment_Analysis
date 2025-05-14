import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

from tokenization import tokenize_words

def loading_data(path):
    file = pd.read_csv(path)
    #file = pd.read_csv("C:\\Users\\arnav\\OneDrive\\Desktop\\Coding Projects\\AI_PROJ_ONE\\IMDB_Dataset.csv")
    x_value = np.where(tokenize_words(file["review"]))
    #y_value = file["sentiment"]
    y_value = np.where(file["sentiment"] == 'positive', 1, 0)

    return x_value, y_value


#x_value = tokenization.make_numerical_vector(x_value)
#print(x_value)
#print(y_value)