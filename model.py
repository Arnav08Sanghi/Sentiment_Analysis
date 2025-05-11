import pandas as pd

file = pd.read_csv("C:\\Users\\arnav\\OneDrive\\Desktop\\Coding Projects\\AI_PROJ_ONE\\IMDB_Dataset.csv")
print(file)

x_value = file["review"]
y_value = file["sentiment"]

print(x_value)
print(y_value)