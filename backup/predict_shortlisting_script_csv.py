import os
import sys
import pandas as pd
from src.prediction.predictor import load_pipeline, predict_shortlisting


# Import Statements:
#     The script starts by importing necessary modules: os, sys, pandas, and the load_pipeline and predict_shortlisting functions from predictor.py in the src.prediction package.
# Function Definitions:
#     load_resume_csv(file_path): This function loads resume data from a CSV file specified by file_path. It uses pd.read_csv from pandas to read the CSV file and returns the data as a DataFrame. If the file is not found, it prints an error message and returns None.
#     main(): This is the main function that orchestrates the script's workflow. It checks the command line arguments, loads CSV files from the specified folder, loads the saved pipeline, predicts shortlisting for each set of resume data, and prints the predicted results.
# Command Line Argument Handling:
#     The script checks if the correct number of command line arguments (one folder path) is provided. If not, it prints a usage message and exits.
#     It verifies if the provided folder path is a valid directory using os.path.isdir. If the directory is not valid, it prints an error message and exits.
# Loading CSV Files:
#     The script uses os.listdir and list comprehension to get a list of CSV files (csv_files) in the specified folder.
#     It then iterates over each CSV file in the folder, loads the resume data using load_resume_csv, and appends the valid data to a list (resumes_data).
# Prediction and Output:
#     After loading all valid resume data, the script loads the saved pipeline (trained_pipeline.pkl) using load_pipeline.
#     It processes each set of resume data in resumes_data, predicts shortlisting using the predict_shortlisting function, and prints the predicted results for each CSV file.
# Usage:
#     To use this script, you would run it from the command line and pass the path to the folder containing CSV files as an argument (python predict_shortlisting_script.py <folder_path>).
#     The script automatically processes all CSV files in the specified folder and predicts the shortlisting status based on the trained model for each set of resumes.
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
        print("Usage: python predict_shortlisting_script.py <folder_path>")
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

    # Load the saved pipeline including the TF-IDF vectorizer and Logistic Regression model
    pipeline = load_pipeline('models/trained_pipeline.pkl')

    # Process each CSV file and predict shortlisting
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        resume_data = load_resume_csv(file_path)
        if resume_data is not None:
            resume_text = resume_data['resume_text'].tolist()
            y_pred = predict_shortlisting(pipeline, resume_text)

            # Print the predicted shortlisting results
            print(f"Resume Shortlisting Predictions from {csv_file}:")
            for text, pred in zip(resume_text, y_pred):
                print(f"Resume: {text}")
                print(f"Shortlisted: {'Yes' if pred else 'No'}")
                print()


if __name__ == "__main__":
    main()
