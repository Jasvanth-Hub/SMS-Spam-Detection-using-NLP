from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from preprocessing import load_and_clean_data, preprocess_data

# Load and preprocess the dataset
file_path = 'sms_dataset.csv'
data = load_and_clean_data(file_path)
data = preprocess_data(data)

# Extract features using CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(data['processed_message']).toarray()

# Extract labels
y = data['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Initialize and train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")


import pickle

# Save the trained model
with open('spam_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the CountVectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)
