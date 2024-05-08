from src.prediction.predictor import load_pipeline, predict_shortlisting


# This script is the entry point for predicting shortlisting for new resume text data.
# It imports the load_pipeline and predict_shortlisting functions from the predictor.py module within the src.prediction package.
# In the main function:
#     a. It loads the saved pipeline (trained_pipeline.pkl) containing the TF-IDF vectorizer and Logistic Regression model.
#     b. Defines example new resume text data.
#     c. Predicts shortlisting for the new resume data using the predict_shortlisting function.
#     d. Prints the predicted shortlisting results for each resume in the example data.
def main():
    # Load the saved pipeline including the TF-IDF vectorizer and Logistic Regression model
    pipeline = load_pipeline('models/trained_pipeline.pkl')

    # Example new resume text data
    new_resume_text = [
        "Energy trader",
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
