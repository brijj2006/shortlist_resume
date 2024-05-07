from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


def train_model(X_train, y_train, max_features=1000):
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    return model, tfidf_vectorizer


def save_pipeline(model, tfidf_vectorizer, file_path):
    pipeline = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('clf', model)
    ])
    joblib.dump(pipeline, file_path)
