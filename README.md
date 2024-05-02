```
# Resume Shortlisting Project

This project implements a machine learning model to shortlist resumes based on predefined criteria. The model uses a logistic regression classifier trained on resume data to predict whether a resume should be shortlisted or not.

## Project Structure

```
```bash
resume_shortlisting_project/
│
├── data/
│   └── resume_dataset.csv          # CSV file containing resume data
│
├── models/
│   └── trained_model.pkl           # Trained machine learning model (saved as a pickle file)
│
├── src/
│   ├── __init__.py
│   ├── shortlisting_model.py       # Python script for training and evaluating the model
│   └── predict_shortlisting.py     # Python script for using the trained model to predict shortlisting for new resumes
│
├── requirements.txt                # File listing project dependencies
└── README.md                       # Project documentation

```

```

- **data/**: Directory containing the dataset file `resume_dataset.csv` containing resume data.
- **models/**: Directory storing the trained machine learning model `trained_model.pkl`.
- **src/**: Source code directory containing scripts for training and using the model.
- **requirements.txt**: File listing project dependencies.
- **README.md**: This file providing project documentation.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/resume-shortlisting-project.git
cd resume-shortlisting-project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python src/shortlisting_model.py
```

4. Predict shortlisting for new resumes:

```bash
python src/predict_shortlisting.py
```

## Dataset

The `resume_dataset.csv` file contains sample resume data for training the model. Each row represents a resume with text content and a label indicating whether the resume was shortlisted or not.
