# 🔥 Burned Calories Predictor

A supervised machine learning web app built with **Streamlit** and **Linear Regression**,
created as part of the Artificial Intelligence course at Damascus University.

## 📌 What it does
- Takes user inputs: gender, age, height, weight, exercise duration, heart rate, and body temperature
- Predicts the number of calories burned using a trained Linear Regression model
- Displays model accuracy metrics (RMSE and R²)

## 🧠 ML Concepts Used
- Supervised Learning
- Linear Regression
- Train/Test Split (80/20)
- RMSE and R-squared evaluation

## 🛠️ Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- NumPy

## 📂 Dataset
- `exercise.csv` — contains user exercise session data
- `calories.csv` — contains calories burned per session
- Both files are merged on `User_ID` during preprocessing

## 🚀 How to Run
1. Clone the repository:
   git clone https://github.com/your-username/calories-predictor.git

2. Install dependencies:
   pip install -r requirements.txt

3. Run the app:
   streamlit run app.py

## 📊 Model Performance
The model is evaluated on a 20% test split.
Metrics are displayed in the app on every run.

## 👤 Author
- **Your Name**
- Damascus University — AI Course, 2024/2025
