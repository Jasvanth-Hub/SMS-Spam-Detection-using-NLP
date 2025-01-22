from preprocessing import load_and_clean_data, preprocess_data

# Load and preprocess dataset
file_path = 'sms_dataset.csv'
data = load_and_clean_data(file_path)
data = preprocess_data(data)

# Display the first few rows
print(data.head())
