import streamlit as st
import pickle


# Load the trained model
with open('spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the CountVectorizer
with open('vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)


# Streamlit title
st.title("SMS Spam Detection System")

st.text("Developed by Jasvanth")

# Text input field
user_input = st.text_area("Enter the SMS text:")

# Button for classification
if st.button("Classify"):
    if user_input.strip():  # Check if input is not empty
        # Preprocess user input
        processed_input = [' '.join(user_input.lower().split())]

        # Transform input into features
        features = cv.transform(processed_input).toarray()

        # Predict using the model
        prediction = model.predict(features)[0]

        # Display results
        if prediction == 1:
            st.error("This is a Spam message.")
        else:
            st.success("This is a Ham message.")
    else:
        st.warning("Please enter a valid SMS message.")
