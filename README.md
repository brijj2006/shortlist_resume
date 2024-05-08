# Resume Shortlisting Model

This project implements a machine learning model for resume shortlisting using text data from resumes. The model is trained using a logistic regression classifier with TF-IDF vectorization for feature extraction.

## Project Structure

The project directory is structured as follows:

- `data/`: Contains the dataset file `resume_dataset_2.csv`.
- `models/`: Stores the trained pipeline (`trained_pipeline.pkl`).
- `src/`: Source code directory.
  - `data_processing/`: Data loading functionality (`data_loader.py`).
  - `model_training/`: Model training modules (`model_trainer.py` and `vectorizer.py`).
  - `prediction/`: Prediction-related modules (`predictor.py` and `predict_shortlisting.py`).
  - `main.py`: Main script for training the model and saving the pipeline.
- `scripts/`: Separate scripts for model training and prediction (`train_model_script.py` and `predict_shortlisting_script.py`).
- `requirements.txt`: File listing the project dependencies.

## Installation

1. Clone this repository to your local machine:
git clone https://github.com/your_username/shortlisting_project.git

css
Copy code

2. Navigate to the project directory:
cd shortlisting_project

arduino
Copy code

3. Create a virtual environment (optional but recommended):
python -m venv venv

r
Copy code

4. Activate the virtual environment:
- On Windows:
  ```
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```
  source venv/bin/activate
  ```

5. Install the required dependencies:
pip install -r requirements.txt

shell
Copy code

## Usage

### Training the Model

To train the model and save the pipeline, run:
python scripts/train_model_script.py

vbnet
Copy code

### Predicting Shortlisting

After training the model, you can predict shortlisting for new resume text data by running:
python scripts/predict_shortlisting_script.py

markdown
Copy code

## Dataset

The dataset used for training the model is `resume_dataset_2.csv` located in the `data/` directory. Ensure that the dataset is appropriately formatted with columns for resume text and shortlisted status.

## Dependencies

The project uses the following Python libraries:
- pandas
- scikit-learn
- joblib

The dependencies are listed in the `requirements.txt` file. You can install them using:
pip install -r requirements.txt