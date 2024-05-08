from src.data_processing.data_loader import load_data
from src.model_training.model_trainer import train_model, save_pipeline


# This script is the entry point for training the shortlisting model and saving the trained pipeline.
# It imports the load_data, train_model, and save_pipeline functions from the respective modules in the src.data_processing and src.model_training packages.
# In the main function:
#     a. It loads the dataset from the CSV file resume_dataset_2.csv.
#     b. Splits the dataset into features (resume text) and labels (shortlisted or not).
#     c. Trains the model using the train_model function, which returns the trained model and TF-IDF vectorizer.
#     d. Saves the trained model and vectorizer as a pipeline using the save_pipeline function.
def main():
    # Load the dataset
    data = load_data('data/resume_dataset_2.csv')

    # Split the dataset into features (resume text) and labels (shortlisted or not)
    X = data['resume_text']
    y = data['shortlisted']

    # Train the model
    model, tfidf_vectorizer = train_model(X, y)

    # Save the trained model
    save_pipeline(model, tfidf_vectorizer, 'models/trained_pipeline.pkl')


if __name__ == "__main__":
    main()
