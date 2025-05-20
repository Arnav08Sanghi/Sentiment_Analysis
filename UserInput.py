from joblib import load

print("SENTIMENT ANALYZER")
user_input = input("Enter your statement here: ")

# Load models and vectorizer (make sure these files exist in your folder)
vectorizer = load("tfidf_vectorizer")  # use the correct filename
X_input = vectorizer.transform([user_input])  # wrap input in a list

# Load models
best_model_rbf = load("rbf_svm_model_C")
svm_model_rbf = load("rbf_svm_model")
svm_model_linear = load("linear_svm_model")

# Predict
Y_value_C_rbf = best_model_rbf.predict(X_input)

# Print sentiment
if Y_value_C_rbf == 1:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")

# Optional: show transformed vector
# print(X_input)