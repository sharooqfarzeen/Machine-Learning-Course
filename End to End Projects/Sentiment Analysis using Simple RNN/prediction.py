import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Importing the pre-saved model
model = tf.keras.models.load_model("model.h5")

# get_words returns a dictioanary of words: index number
word_index = imdb.get_word_index()

# maximum length of a review
max_length = 500

# Function to pre-process the movie
def pre_process(review):
    # Lower case
    review = review.lower()
    
    # Word to index
    encoded_review = [word_index.get(word, 2) + 3 for word in review]
    
    # Padding
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_length, padding='pre')
    
    return padded_review

# Function to predict sentiment
def predict(review):

    score = model.predict(pre_process(review))

    if score > 0.5:
        return "Review is positive.", score
    
    return "Review is negative.", score