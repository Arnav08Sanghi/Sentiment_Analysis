import joblib 

print("SENTIMENT ANALYZER")
user_input = input("Enter your statement here: ")

# Load models and vectorizer (make sure these files exist in your folder)
vectorizer = joblib.load("C:\\Users\\arnav\\OneDrive\\Desktop\\Coding Projects\\AI_PROJ_ONE\\tfidf_vectorizer.pkl")  # use the correct filename
X_input = vectorizer.transform([user_input])  # wrap input in a list

# Load models
#best_model_rbf = load("rbf_svm_model_C")
svm_model_rbf = joblib.load("C:\\Users\\arnav\\OneDrive\\Desktop\\Coding Projects\\AI_PROJ_ONE\\rbf_svm_model.pkl")
svm_model_linear = joblib.load("C:\\Users\\arnav\\OneDrive\\Desktop\\Coding Projects\\AI_PROJ_ONE\\linear_svm_model.pkl")

# Predict
Y_value_C_rbf = svm_model_rbf.predict(X_input)

# Print sentiment
if Y_value_C_rbf == 1:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")
