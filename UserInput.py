from joblib import load

print("SENTIMENT ANALYZIER")
user_input = input("Enter your statement here: ")

vectorizer = load("tfidf_vectorizer")

X_input = vectorizer.transform(user_input)

print(X_input)