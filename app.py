import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import numpy as np

@st.cache_resource
def train_model():
    st.info("Loading, processing, and training the linear regression model... (Please wait a moment)")

    try:
        exercise_data = pd.read_csv('exercise.csv')
        calories_data = pd.read_csv('calories.csv')
        data = pd.merge(exercise_data, calories_data, on='User_ID')
    except FileNotFoundError:
        st.error("Error: Data files not found.")
        return None

    data = data.drop('User_ID', axis=1)
    data['Gender'] = data['Gender'].astype(str).str.lower()
    data['Gender'].fillna('male', inplace=True)
    data['Gender'] = data['Gender'].map({'male': 0, 'female': 1})
    data = data.fillna(data.mean(numeric_only=True))

    X = data[['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
    y = data['Calories']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.success("Model training completed successfully.")
    st.info("Model accuracy metrics on test data:")
    st.code(f"RMSE (Mean Error): {rmse:.2f} | R-squared (Goodness of Fit): {r2:.2f}")
    return model

# Run training
model = train_model()
if model is not None:
    st.title("Burned Calories Predictor (Supervised Regression)")
    st.markdown("---")
    st.sidebar.header("Enter User & Exercise Details")

    gender_input = st.sidebar.radio("Gender", ('Male', 'Female'))
    gender_encoded = 0 if gender_input == 'Male' else 1  # Use 0 and 1 for prediction

    age = st.sidebar.slider("Age (years)", 15, 80, 30)
    height = st.sidebar.slider("Height (cm)", 140.0, 200.0, 170.0)
    weight = st.sidebar.slider("Weight (kg)", 40.0, 120.0, 70.0)
    duration = st.sidebar.slider("Exercise Duration (minutes)", 5, 90, 30)
    heart_rate = st.sidebar.slider("Heart Rate (beats/min)", 70, 130, 100)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 36.0, 41.0, 37.5)

    input_data = pd.DataFrame({
        'Gender': [gender_encoded],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'Duration': [duration],
        'Heart_Rate': [heart_rate],
        'Body_Temp': [body_temp]
    })

    if st.sidebar.button('Predict Calories'):
        prediction = model.predict(input_data)
        st.subheader("Prediction Result:")
        calories_burnt = int(np.round(prediction[0]))
        st.success("Estimated calories burned:")
        st.markdown(f"## {calories_burnt} Calories")
        st.balloons()
