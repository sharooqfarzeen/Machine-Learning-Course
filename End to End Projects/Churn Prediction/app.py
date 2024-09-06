import pandas as pd
import streamlit as st
import pickle
import tensorflow as tf

# Loading our pre-saved model
model = tf.keras.models.load_model('model_1.keras')

# Loading data preprocessor
with open('preprocessor.pkl', 'rb') as obj:
    preprocessor = pickle.load(obj)

# Retrieve OneHotEncoder from ColumnTransformer
one_hot_encoder = preprocessor.named_transformers_['cat']
gender_list = one_hot_encoder.categories_[1]
country_list = one_hot_encoder.categories_[0]

## Streamlit app
st.title('Customer Churn Prediction')

# Receiving data from the user
geography = st.selectbox('Geography', country_list)
gender = st.selectbox('Gender', gender_list)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'Geography': [geography],
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Debug: Print input data
st.write("Input Data:", input_data)

# Transform input data
new_data = preprocessor.transform(input_data)

# Debug: Print transformed data
st.write("Transformed Data:", new_data)

# Predict churn
prediction = model.predict(new_data)

# Debug: Print prediction
st.write("Prediction:", prediction)

prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')