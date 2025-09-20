import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib


# Load the trained model
model = joblib.load("rainfall_model.pkl")

st.title("Rainfall Prediction App")
st.write("Predict whether it will rain tomorrow using Machine Learning!")

# Sidebar input form
st.sidebar.header("Input Features")

def user_input():
    # Real features from your dataset
    pressure = st.sidebar.number_input("Pressure (hPa)", 800.0, 1100.0, 1013.0)
    dewpoint = st.sidebar.number_input("Dew Point (°C)", -10.0, 35.0, 15.0)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)
    cloud = st.sidebar.slider("Cloud Cover (%)", 0, 100, 40)
    
    sunshine = st.sidebar.number_input("Sunshine (hours)", 0.0, 15.0, 7.0)
    winddirection = st.sidebar.number_input("Wind Direction (°)", 0.0, 360.0, 30.0)
    
    windspeed = st.sidebar.number_input("Wind Speed (km/h)", 0, 150, 20)

    # Convert categorical winddirection into numeric (One-Hot Encoding placeholder)
    data = {
        "pressure": pressure,
        "dewpoint": dewpoint,
        "humidity": humidity,
        "cloud": cloud,
         
        "sunshine": sunshine,
        "winddirection": winddirection,
        "windspeed": windspeed
    }

    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input()

# Show input
st.subheader("User Input Parameters")
st.write(input_df)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    result = "Yes, it will rain tomorrow!" if prediction[0] == 1 else "☀ No, it will not rain tomorrow."
    st.subheader("Prediction Result")
    st.success(result)
