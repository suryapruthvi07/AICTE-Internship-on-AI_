import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

import warnings
warnings.filterwarnings('ignore')

st.title("Personal Fitness Tracker")
st.write("This WebApp predicts the calories burned based on parameters like Age, BMI, Duration, Heart Rate, Body Temperature, and Gender.")

# Sidebar for User Inputs
st.sidebar.header("User Input Parameters")
def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 20.0)
    duration = st.sidebar.slider("Duration (min)", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 36.0, 42.0, 38.0)
    gender = st.sidebar.radio("Gender", ["Male", "Female"])

    data = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": 1 if gender == "Male" else 0
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()
st.write("### Your Input Parameters")
st.write(df)

# Load Data
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")
    df["BMI"] = round(df["Weight"] / (df["Height"] / 100) ** 2, 2)
    df = pd.get_dummies(df, columns=["Gender"], drop_first=True)
    return df

data = load_data()

# Train-Test Split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)
X_train, y_train = train_data.drop("Calories", axis=1), train_data["Calories"]
X_test, y_test = test_data.drop("Calories", axis=1), test_data["Calories"]

# Model Training
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6, random_state=1)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Prediction
st.write("### Prediction")
df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = model.predict(df)

st.success(f"Predicted Calories Burned: {round(prediction[0], 2)} kcal")

# Display Similar Results
st.write("### Similar Results")
similar_data = data[(data['Calories'] >= prediction[0] - 10) & (data['Calories'] <= prediction[0] + 10)]
st.write(similar_data.sample(min(5, len(similar_data))))

# General Comparisons
st.write("### General Comparison")
st.write(f"You are older than {np.mean(data['Age'] < df['Age'].iloc[0]) * 100:.2f}% of others.")
st.write(f"Your exercise duration is higher than {np.mean(data['Duration'] < df['Duration'].iloc[0]) * 100:.2f}% of others.")
st.write(f"Your heart rate is higher than {np.mean(data['Heart_Rate'] < df['Heart_Rate'].iloc[0]) * 100:.2f}% of others.")
st.write(f"Your body temperature is higher than {np.mean(data['Body_Temp'] < df['Body_Temp'].iloc[0]) * 100:.2f}% of others.")
