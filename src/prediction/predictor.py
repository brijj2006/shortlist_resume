import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


# This script contains two functions: load_pipeline and predict_shortlisting.
# load_pipeline loads a saved pipeline (containing a TF-IDF vectorizer and a logistic regression model) from a file using joblib's load function.
#     a. It handles the case where the pipeline file is not found by raising a FileNotFoundError and printing an error message before exiting.
#     b. It returns the loaded pipeline.
def load_pipeline(file_path):
    try:
        pipeline = joblib.load(file_path)
    except FileNotFoundError as e:
        print('Pipeline file not found. Please provide the correct file path.')
        print('Exception:', e)
        exit()
    return pipeline


# predict_shortlisting takes a loaded pipeline and new resume text data as input and predicts shortlisting for the new resumes.
#     a. It transforms the new resume text data into TF-IDF features using the TF-IDF vectorizer in the pipeline (pipeline.named_steps['tfidf'].transform).
#     b. It predicts shortlisting using the logistic regression model in the pipeline (pipeline.named_steps['clf'].predict) and returns the predictions.
def predict_shortlisting(pipeline, new_resume_text):
    X_new_tfidf = pipeline.named_steps['tfidf'].transform(new_resume_text)
    return pipeline.named_steps['clf'].predict(X_new_tfidf)
