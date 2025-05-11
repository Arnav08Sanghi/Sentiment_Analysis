import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tokenization

file = pd.read_csv("C:\\Users\\arnav\\OneDrive\\Desktop\\Coding Projects\\AI_PROJ_ONE\\IMDB_Dataset.csv")
print(file)

x_value = file["review"]
y_value = file["sentiment"]


x_value['text'] = tokenization.make_numerical_vector(x_value['text'])

y_value['text'] = np.where(y_value['text'] == 'positive', '1', '0')


print(x_value)
print(y_value)