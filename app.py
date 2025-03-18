import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Title of the app
st.title(" Fitness Tracker ")
st.write("This app predicts calorie burn based on user input.")

# Sidebar for user input
st.sidebar.header("Enter Your Details")

def user_inputs():
    age = st.sidebar.slider("Age", 10, 80, 25)
    weight = st.sidebar.slider("Weight (kg)", 30, 150, 70)
    height = st.sidebar.slider("Height (cm)", 100, 220, 170)
    duration = st.sidebar.slider("Exercise Duration (min)", 5, 60, 30)
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 50, 180, 100)
    return np.array([[age, weight, height, duration, heart_rate]])

# Load dataset
data = pd.read_csv("calories.csv")

# Preprocess data
X = data[['Age', 'Weight', 'Height', 'Duration', 'Heart_Rate']]
y = data['Calories']

# ✅ Define scaler BEFORE using it
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit the scaler with data

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get user input
user_data = user_inputs()

# ✅ Convert user input into DataFrame (this ensures column names match)
user_data_df = pd.DataFrame(user_data, columns=X.columns)

# ✅ Scale user input (Now, scaler is correctly defined)
user_data_scaled = scaler.transform(user_data_df)

# Predict calories
predicted_calories = model.predict(user_data_scaled)

st.subheader("Predicted Calories Burned")
st.write(f"{predicted_calories[0]:.2f} kcal")

# Display data insights
st.subheader("Data Insights")
st.write("Sample Data:")
st.dataframe(data.head())

# Plot
fig, ax = plt.subplots()
sns.scatterplot(x=data['Duration'], y=data['Calories'], ax=ax)
ax.set_xlabel("Exercise Duration (min)")
ax.set_ylabel("Calories Burned")
st.pyplot(fig)
