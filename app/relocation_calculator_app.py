import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Set page configuration
st.set_page_config(page_title="Relocation Cost Predictor", page_icon=":house:", layout="wide")

# Load CSS styles
with open(os.path.join(script_dir, "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Streamlit app
st.sidebar.title("Relocation Cost Predictor")
st.sidebar.image(os.path.join(script_dir, "logo.png"), use_column_width=True)

# Display app description and instructions
st.sidebar.markdown("""
This app is a proof of concept that predicts the relocation cost based on various factors such as the number of stairs, objects to dismantle and pack, distance to the moving vehicle, moving duration, number of movers, and effort level.

Please note that the predictions are based on hypothetical values and do not represent real-world costs. This app is intended for demonstrative purposes only.

To use the app:
1. Adjust the input parameters using the sliders and dropdown in the sidebar.
2. Click the "Predict" button to get the estimated relocation cost.
3. View the cost breakdown and feature importance for insights.
""")

# Generate a fictional dataset
np.random.seed(42)
num_samples = 1000
initial_stairs = np.random.choice([0, 1, 2, 3], size=num_samples)
destination_stairs = np.random.choice([0, 1, 2, 3], size=num_samples)
objects_to_dismantle = np.random.randint(0, 10, size=num_samples)
objects_to_pack = np.random.randint(10, 100, size=num_samples)
distance_to_vehicle = np.random.randint(10, 100, size=num_samples)
moving_duration = np.random.randint(2, 8, size=num_samples)
num_movers = np.random.randint(2, 6, size=num_samples)
effort_level = np.random.choice(['Low', 'Medium', 'High'], size=num_samples)
relocation_cost = 500 + (200 * initial_stairs) + (200 * destination_stairs) + (50 * objects_to_dismantle) + (10 * objects_to_pack) + (5 * distance_to_vehicle) + (100 * moving_duration) + (200 * num_movers) + (300 * (effort_level == 'Medium')) + (500 * (effort_level == 'High')) + np.random.normal(0, 200, size=num_samples)

# Create a DataFrame
data = pd.DataFrame({
    'Initial Stairs': initial_stairs,
    'Destination Stairs': destination_stairs,
    'Objects to Dismantle': objects_to_dismantle,
    'Objects to Pack': objects_to_pack,
    'Distance to Vehicle (feet)': distance_to_vehicle,
    'Moving Duration (hours)': moving_duration,
    'Number of Movers': num_movers,
    'Effort Level': effort_level,
    'Relocation Cost ($)': relocation_cost
})

# Encode categorical variables
data = pd.get_dummies(data, columns=['Effort Level'])

# Input features
initial_stairs = st.sidebar.slider("Initial Stairs", int(data['Initial Stairs'].min()), int(data['Initial Stairs'].max()), int(data['Initial Stairs'].mean()))
destination_stairs = st.sidebar.slider("Destination Stairs", int(data['Destination Stairs'].min()), int(data['Destination Stairs'].max()), int(data['Destination Stairs'].mean()))
objects_to_dismantle = st.sidebar.slider("Objects to Dismantle", int(data['Objects to Dismantle'].min()), int(data['Objects to Dismantle'].max()), int(data['Objects to Dismantle'].mean()))
objects_to_pack = st.sidebar.slider("Objects to Pack", int(data['Objects to Pack'].min()), int(data['Objects to Pack'].max()), int(data['Objects to Pack'].mean()))
distance_to_vehicle = st.sidebar.slider("Distance to Vehicle (feet)", int(data['Distance to Vehicle (feet)'].min()), int(data['Distance to Vehicle (feet)'].max()), int(data['Distance to Vehicle (feet)'].mean()))
moving_duration = st.sidebar.slider("Moving Duration (hours)", int(data['Moving Duration (hours)'].min()), int(data['Moving Duration (hours)'].max()), int(data['Moving Duration (hours)'].mean()))
num_movers = st.sidebar.slider("Number of Movers", int(data['Number of Movers'].min()), int(data['Number of Movers'].max()), int(data['Number of Movers'].mean()))
effort_level = st.sidebar.selectbox("Effort Level", ['Low', 'Medium', 'High'])

# Prepare the features and target
X = data.drop('Relocation Cost ($)', axis=1)
y = data['Relocation Cost ($)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Prepare the user input features
input_data = pd.DataFrame({
    'Initial Stairs': [initial_stairs],
    'Destination Stairs': [destination_stairs],
    'Objects to Dismantle': [objects_to_dismantle],
    'Objects to Pack': [objects_to_pack],
    'Distance to Vehicle (feet)': [distance_to_vehicle],
    'Moving Duration (hours)': [moving_duration],
    'Number of Movers': [num_movers],
    'Effort Level_Low': [1 if effort_level == 'Low' else 0],
    'Effort Level_Medium': [1 if effort_level == 'Medium' else 0],
    'Effort Level_High': [1 if effort_level == 'High' else 0]
}, columns=X.columns)

# Predict the relocation cost based on user input
if st.sidebar.button("Predict"):
    predicted_cost = model.predict(input_data)[0]

    # Display the prediction
    st.header("Predicted Relocation Cost")
    st.subheader(f"${predicted_cost:.2f}")

    # Show cost breakdown
    st.subheader("Cost Breakdown")
    cost_breakdown = {
        "Base Cost": 500,
        "Initial Stairs": initial_stairs * 200,
        "Destination Stairs": destination_stairs * 200,
        "Objects to Dismantle": objects_to_dismantle * 50,
        "Objects to Pack": objects_to_pack * 10,
        "Distance to Vehicle": distance_to_vehicle * 5,
        "Moving Duration": moving_duration * 100,
        "Number of Movers": num_movers * 200,
        "Effort Level": 300 if effort_level == 'Medium' else 500 if effort_level == 'High' else 0
    }
    breakdown_df = pd.DataFrame.from_dict(cost_breakdown, orient='index', columns=['Cost ($)'])
    st.table(breakdown_df)

    # Display evaluation metrics
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
    col2.metric("R-squared (R2) Score", f"{r2:.2f}")

    # Visualize the feature importance
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    st.pyplot(fig)

# Display dataset info
with st.expander("Dataset Info"):
    st.write(data.head())
    st.write(f"Dataset size: {data.shape[0]} samples")