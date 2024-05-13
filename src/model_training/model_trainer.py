from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


# This script contains two functions: train_model and save_pipeline.
# train_model takes training data X_train and labels y_train as input and trains a logistic regression model using TF-IDF vectorization.
#   a. It creates a TF-IDF vectorizer (TfidfVectorizer) with a specified maximum number of features (max_features).
#   b. It transforms the training data X_train into TF-IDF features using fit_transform.
#   c. It trains a logistic regression model (LogisticRegression) on the TF-IDF transformed data and returns the trained model and TF-IDF vectorizer.
def train_model(X, y, max_features=1000):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print('Training Set Class Distribution:')
    print(y_train.value_counts())

    print('\nTesting Set Class Distribution:')
    print(y_test.value_counts())

    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Predict shortlisting for test resumes
    y_pred = model.predict(X_test_tfidf)

    # Calculate and print accuracy details
    accuracy = accuracy_score(y_test, y_pred)
    print('\nAccuracy:', accuracy)

    # Print Classification report (includes precision, recall, fi-score and support)
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, tfidf_vectorizer


# save_pipeline takes a trained model, TF-IDF vectorizer, and a file path as input and saves them as a pipeline using joblib's dump function.
#   a. It creates a pipeline (Pipeline) consisting of the TF-IDF vectorizer and the trained model.
#   b. It saves the pipeline to the specified file path using joblib.dump.
def save_pipeline(model, tfidf_vectorizer, file_path):
    pipeline = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('clf', model)
    ])
    joblib.dump(pipeline, file_path)
