from src.prediction.predictor import load_pipeline, predict_shortlisting


def main():
    # Load the saved pipeline including the TF-IDF vectorizer and Logistic Regression model
    pipeline = load_pipeline('models/trained_pipeline.pkl')

    # Example new resume text data
    new_resume_text = [
        "Experienced software engineer with expertise in Python and machine learning.",
        "Customer-oriented sales professional with a track record of exceeding targets."
    ]

    # Predict shortlisting for the new resume data
    y_pred_new = predict_shortlisting(pipeline, new_resume_text)

    # Print the predicted shortlisting results
    for text, pred in zip(new_resume_text, y_pred_new):
        print(f"Resume: {text}")
        print(f"Shortlisted: {'Yes' if pred else 'No'}")
        print()


if __name__ == "__main__":
    main()
