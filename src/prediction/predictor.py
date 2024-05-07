import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def load_pipeline(file_path):
    try:
        pipeline = joblib.load(file_path)
    except FileNotFoundError as e:
        print('Pipeline file not found. Please provide the correct file path.')
        print('Exception:', e)
        exit()
    return pipeline


def predict_shortlisting(pipeline, new_resume_text):
    X_new_tfidf = pipeline.named_steps['tfidf'].transform(new_resume_text)
    return pipeline.named_steps['clf'].predict(X_new_tfidf)
