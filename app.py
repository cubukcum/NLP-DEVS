import streamlit as st
import requests

# Streamlit App Title
st.title("Product Evaluation")

# Input field for user question
question = st.text_input("Enter your question:", "What is the overall evaluation of the product?")

# Button to submit the question
if st.button("Evaluate"):
    # Send a request to the Flask API
    try:
        response = requests.post(
            "http://localhost:5000/analyze_product",  # Ensure this matches your Flask app's URL
            json={"question": question, "csv_file_path": "reviews1.csv"}  # Using POST with JSON payload
        )

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            print(data)
            st.subheader("Response:")
            st.write(data.get('response', 'No response received'))
        else:
            st.error(f"Error fetching response from API. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        
        
review = st.text_area("Enter the review text:", "")

# Button to submit the review
if st.button("Classify Review"):
    # Send a request to the Flask API
    try:
        response = requests.post(
            "http://localhost:5000/classify_review",  # Ensure this matches your Flask app's URL
            json={"review": review}  # Using POST with JSON payload
        )

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            st.subheader("Classification Result:")
            st.write(f"Sentiment: {data.get('sentiment')}")
            st.write(f"Review Details: {', '.join(data.get('review_details', []))}")
            st.write(f"Confidence: {data.get('confidence')}")
            st.write(f"Summary: {', '.join(data.get('summary', []))}")
            st.write(f"Suggestions: {data.get('suggestions')}")
        else:
            st.error(f"Error fetching response from API. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")