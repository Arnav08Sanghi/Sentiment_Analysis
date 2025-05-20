from joblib import load

print("SENTIMENT ANALYZIER")
user_input = input("Enter your statement here: ")

vectorizer = load("tfidf_vectorizer")

X_input = vectorizer.transform(user_input)


best_model_rbf = load("rbf_svm_model_C")
svm_model_rbf = load("rbf_svm_model")
svm_model_linear = load("linear_svm_model")

print(X_input)