from src.data_processing.data_loader import load_data
from src.model_training.model_trainer import train_model, save_pipeline


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
