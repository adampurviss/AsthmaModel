import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import tkinter as tk
from tkinter import filedialog


root = tk.Tk()
root.withdraw()

# Open file dialog to select a CSV file
file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
root.destroy()

df = pd.read_csv(file_path)
print("Columns in the dataset:", df.columns.tolist())

# Define the categorical and numerical features in the dataset
categorical_features = ['gender', 'air_quality', 'living_conditions', 'location']
numerical_features = ['age', 'temperature', 'humidity', 'fev1', 'bmi']

# Create a ColumnTransformer to preprocess numerical and categorical features separately
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False), categorical_features)
    ])


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Split the data into features (X) and target variable (y)
X = df.drop(columns=['asthma_diagnosed', 'patient_id'])
y = df['asthma_diagnosed']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using the training data
pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model pipeline to a file for later use
joblib.dump(pipeline, 'asthma_model_pipeline.pkl')
print("Model pipeline saved successfully.")
