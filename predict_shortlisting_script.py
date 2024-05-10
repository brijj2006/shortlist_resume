import os
import sys
from PyPDF2 import PdfReader  # Import PdfReader from PyPDF2
from src.prediction.predictor import load_pipeline, predict_shortlisting

def extract_text_from_pdf(file_path):
    try:
        # Initialize a PDF reader object
        pdf_reader = PdfReader(file_path)

        # Extract text from all pages and concatenate
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

        # Remove newline characters and extra spaces
        text = ' '.join(text.split())
    except Exception as e:
        print(f'Error extracting text from PDF: {e}')
        return None
    return text

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_shortlisting_script.py <folder_path>")
        exit()

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        exit()

    # Get list of PDF files in the specified folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in {folder_path}.")
        exit()

    # Load the saved pipeline including the TF-IDF vectorizer and Logistic Regression model
    pipeline = load_pipeline('models/trained_pipeline.pkl')

    # Process each PDF file and predict shortlisting
    for pdf_file in pdf_files:
        file_path = os.path.join(folder_path, pdf_file)
        resume_text = extract_text_from_pdf(file_path)
        if resume_text is not None:
            y_pred = predict_shortlisting(pipeline, [resume_text])

            # Print the predicted shortlisting result
            print(f"Resume from {pdf_file}: {resume_text}")
            print(f"Shortlisted: {'Yes' if y_pred[0] else 'No'}")
            print()

if __name__ == "__main__":
    main()
