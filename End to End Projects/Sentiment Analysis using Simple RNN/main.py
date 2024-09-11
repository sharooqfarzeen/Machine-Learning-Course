import tensorflow as tf
import streamlit as st

from prediction import predict

# Title for the app
st.title("Review Sentiment Analysis")
# Take user input
review = st.text_input(label="Review", placeholder="Please enter your review")

# Prediction
if st.button('Analyze'):
    
    sentiment, score = predict(review)
    
    # Display the result
    st.write(sentiment)
    st.write(f'Prediction Score: {score[0][0]:.2f}')
else:
    st.write('Please enter a movie review.')