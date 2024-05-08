import os
import sys
import pandas as pd
from src.data_processing.data_loader import load_data
from src.model_training.model_trainer import train_model, save_pipeline


# The main function checks if the correct number of command line arguments (one folder path) is provided. If not, it prints a usage message and exits.
# It verifies if the provided folder path is a valid directory using os.path.isdir. If the directory is not valid, it prints an error message and exits.
# It gets a list of CSV files (csv_files) in the specified folder using a list comprehension and os.listdir.
# It loads and concatenates data from each CSV file into a single DataFrame using the load_resume_csv function and pd.concat.
# It then splits the combined dataset into features (resume text) and labels (shortlisted or not), trains the model using the train_model function, and saves the trained model and TF-IDF vectorizer as a pipeline using save_pipeline.
# To use this script, you would run it from the command line and pass the path to the folder containing the CSV files as an argument. For example:
#     python train_model_script.py path/to/folder_containing_csv_files
def load_resume_csv(file_path):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError as e:
        print(f'File not found: {file_path}. Please provide the correct file path.')
        print('Exception:', e)
        return None
    return data


def main():
    if len(sys.argv) != 2:
        print("Usage: python train_model_script.py <folder_path>")
        exit()

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        exit()

    # Get list of CSV files in the specified folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in {folder_path}.")
        exit()

    # Load and concatenate data from each CSV file
    all_data = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        resume_data = load_resume_csv(file_path)
        if resume_data is not None:
            all_data.append(resume_data)

    if not all_data:
        print("No valid data found in CSV files.")
        exit()

    # Concatenate all data into a single DataFrame
    data = pd.concat(all_data, ignore_index=True)

    # Split the dataset into features (resume text) and labels (shortlisted or not)
    X = data['resume_text']
    y = data['shortlisted']

    # Train the model
    model, tfidf_vectorizer = train_model(X, y)

    # Save the trained model and vectorizer as a pipeline
    save_pipeline(model, tfidf_vectorizer, 'models/trained_pipeline.pkl')


if __name__ == "__main__":
    main()
