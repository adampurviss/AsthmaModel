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

file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
root.destroy()

data_frame = pd.read_csv(file_path)

categorical_features = ['gender', 'air_quality', 'living_conditions', 'location']
numerical_features = ['age', 'temperature', 'humidity', 'fev1', 'bmi']

# transforms data into machine learning friendly data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False), categorical_features)
    ])


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # This code will limit itself to 1000 tries trying to get the most optimal model
    ('classifier', LogisticRegression(max_iter=1000))
])

features = data_frame.drop(columns=['asthma_diagnosed', 'patient_id'])
target_variable = data_frame['asthma_diagnosed']

# The model splits the data and will randomly assign which it will train from, and test against.
# Test_size=0.2 reserves 20% of the data for testing.
# The random state ensures consistency to make results reproducible
features_train, features_test, target_variable_train, target_variable_test = train_test_split(features, target_variable, test_size=0.2, random_state=42)

# Pipeline.fit applies the preprocessing steps and trains the classifier.
pipeline.fit(features_train, target_variable_train)

# Test model against reserved test data
target_variable_prediction = pipeline.predict(features_test)

accuracy = accuracy_score(target_variable_test, target_variable_prediction)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model pipeline to a file for later use
joblib.dump(pipeline, 'asthma_model_pipeline.pkl')
print("Model pipeline saved successfully.")

