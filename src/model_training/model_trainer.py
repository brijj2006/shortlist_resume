from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


# This script contains two functions: train_model and save_pipeline.
# train_model takes training data X_train and labels y_train as input and trains a logistic regression model using TF-IDF vectorization.
#   a. It creates a TF-IDF vectorizer (TfidfVectorizer) with a specified maximum number of features (max_features).
#   b. It transforms the training data X_train into TF-IDF features using fit_transform.
#   c. It trains a logistic regression model (LogisticRegression) on the TF-IDF transformed data and returns the trained model and TF-IDF vectorizer.
def train_model(X_train, y_train, max_features=1000):
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

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
