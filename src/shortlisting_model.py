import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
data = pd.read_csv('data/resume_dataset.csv')

# Split the dataset into features (resume text) and labels (shortlisted or not)
X = data['resume_text']
y = data['shortlisted']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF vectorizer to convert resume text into numerical features
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust the number of features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save the trained model
joblib.dump(model, 'models/trained_model.pkl')

# Predict shortlisting for test resumes
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print(classification_report(y_test, y_pred))
