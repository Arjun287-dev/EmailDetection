import streamlit as st
import pandas as pd
import numpy as np
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Set page configuration
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide"
)

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')

download_nltk_resources()

# Initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Define preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    try:
        # Try to load the dataset - adjust the path if necessary
        df = pd.read_csv('emailDataset.csv')
        data = df.where((pd.notnull(df)), '')
        
        # Convert label to numeric
        data.loc[data['v1'] == 'spam', 'v1'] = 0
        data.loc[data['v1'] == 'ham', 'v1'] = 1
        
        # Add cleaned text
        data['cleaned'] = data['v2'].apply(preprocess_text)
        
        # Split data
        X = data['v1'].astype(int)
        Y = data['v2']
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
        
        # Vectorize data
        tfidf = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
        Y_train_tfidf = tfidf.fit_transform(Y_train)
        Y_test_tfidf = tfidf.transform(Y_test)
        
        # Train model
        model = LogisticRegression()
        model.fit(Y_train_tfidf, X_train)
        
        # Calculate accuracy
        train_pred = model.predict(Y_train_tfidf)
        train_accuracy = accuracy_score(X_train, train_pred)
        
        test_pred = model.predict(Y_test_tfidf)
        test_accuracy = accuracy_score(X_test, test_pred)
        
        return model, tfidf, train_accuracy, test_accuracy, True
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, 0, 0, False

# Main application
def main():
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses machine learning to detect spam emails. "
        "Enter the content of an email to check if it's spam or not."
    )
    
    st.sidebar.title("Model Info")
    model, tfidf, train_accuracy, test_accuracy, model_loaded = load_and_preprocess_data()
    
    if model_loaded:
        st.sidebar.success("Logistic Regression Model loaded successfully!")
        st.sidebar.metric("Training Accuracy", f"{train_accuracy:.2%}")
        st.sidebar.metric("Testing Accuracy", f"{test_accuracy:.2%}")
    else:
        st.sidebar.error("Failed to load model. Check if dataset file exists.")
    
    # Main content
    st.title("📧 Spam Email Detector")
    st.write("""
    ### Enter email content to check if it's spam or not
    This tool analyzes the content of emails to determine if they're legitimate (ham) or spam.
    """)
    
    # Input area for email content
    email_content = st.text_area(
        "Email Content", 
        height=200,
        placeholder="Enter the subject and body of the email you want to analyze..."
    )
    
    # Classification
    if st.button("Analyze Email"):
        if not email_content:
            st.warning("Please enter some email content to analyze.")
        elif not model_loaded:
            st.error("Model not loaded. Cannot analyze email.")
        else:
            # Preprocess and predict
            input_data = tfidf.transform([email_content])
            prediction = model.predict(input_data)[0]
            
            # Display results
            st.subheader("Analysis Result:")
            
            if prediction == 1:
                st.success("✅ This is likely a legitimate email (Ham)")
                probability = model.predict_proba(input_data)[0][1]
                st.progress(probability)
                st.write(f"Confidence: {probability:.2%}")
            else:
                st.error("🚨 This appears to be SPAM")
                probability = model.predict_proba(input_data)[0][0] 
                st.progress(probability)
                st.write(f"Confidence: {probability:.2%}")
            
            # Show some spam indicators
            if prediction == 0:
                st.subheader("Potential Spam Indicators:")
                st.write("""
                - Urgency language or limited time offers
                - Requests for personal information
                - Poor grammar or unusual formatting
                - Promises of money or prizes
                - Suspicious links or attachments
                """)
    
if __name__ == "__main__":
    main()