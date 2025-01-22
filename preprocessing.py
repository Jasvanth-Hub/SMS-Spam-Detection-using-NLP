import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()

def preprocess_text(text):
    # Remove special characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()

    # Tokenize, remove stopwords, and apply stemming
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]

    return ' '.join(text)

def preprocess_data(data):
    data['processed_message'] = data['message'].apply(preprocess_text)
    return data



import pandas as pd

def load_and_clean_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path, encoding='latin-1')

    # Drop unnecessary columns
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']

    # Encode labels (spam=1, ham=0)
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    # Handle null values
    data.dropna(inplace=True)
    return data
