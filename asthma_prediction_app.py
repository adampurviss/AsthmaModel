#asthma_prediction_application

import pandas as pd
import tkinter as tk
from tkinter import filedialog, Text, ttk
import joblib

pipeline = joblib.load('asthma_model_pipeline.pkl')

factor_weights = {
    'family_history': 0.6,
    'smoker': 0.2,
    'overweight_obese': 0.2,
    'cold_temperature': 0.05,
    'poor_air_quality': 0.2, # TO DO EVALUATE
    'living_conditions': {
        'Urban': 0.2,
        'Suburban': 0.1,
        'Rural': 0
    }
}


def preprocess_data(data_frame):
    # Removes asthma diagnosed column
    data_frame = data_frame.drop(columns=['asthma_diagnosed'])
    return data_frame


def predict_asthma(data_frame):
    preprocessed_data = preprocess_data(data_frame)
    # Using pipeline, predicting probability of preprocessed data
    # It selects all rows and gets the probability of them having asthma
    probabilities = pipeline.predict_proba(preprocessed_data)[:, 1]
    return probabilities


def calculate_risk_score(row):
    # This accesses the data for each patient, and calculates their risk score
    score = 0
    if row['asthma_within_family'] == 1:
        score += factor_weights['family_history']
    if row['smoker'] == 1:
        score += factor_weights['smoker']
    if row['bmi'] > 25:
        score += factor_weights['overweight_obese']
    if row['temperature'] < 15:
        score += factor_weights['cold_temperature']
    if row['air_quality'] in ['Poor', 'Very Poor']:
        score += factor_weights['poor_air_quality']
    score += factor_weights['living_conditions'].get(row['living_conditions'], 0)
    return score


def categorise_risk(score):
    if score >= 0.9:
        return 'Very High Risk'
    elif score >= 0.7:
        return 'High Risk'
    elif score >= 0.5:
        return 'Medium Risk'
    elif score >= 0.3:
        return 'Low Risk'
    else:
        return 'Very Low Risk'


def potentially_undiagnosed(row, fev1_threshold=2.5):
    if row['risk_category'] == 'Very High Risk' and row['fev1'] < fev1_threshold:
        return 'Potentially Undiagnosed Asthma'
    else:
        return row['risk_category']


def generate_prediction_report(filters, risk_group):
    # Load data from CSV files into a list of DataFrames
    data_frames = [pd.read_csv(file) for file in file_paths]

    # Combine all DataFrames into one
    combined_data = pd.concat(data_frames, ignore_index=True)

    # If filter is applied then do what filter states, if not, don't.
    if filters['age_min'] is not None and filters['age_max'] is not None:
        combined_data = combined_data[
            (combined_data['age'] >= filters['age_min']) &
            (combined_data['age'] <= filters['age_max'])
            ]
    if filters['gender'] != "All":
        combined_data = combined_data[combined_data['gender'] == filters['gender']]
    if filters['location'] != "All":
        combined_data = combined_data[combined_data['location'] == filters['location']]

    undiagnosed_data = combined_data[combined_data['asthma_diagnosed'] == 0].copy()
    # if there is people without asthma, then calculate risk score and category
    if not undiagnosed_data.empty:
        undiagnosed_data['risk_score'] = undiagnosed_data.apply(calculate_risk_score, axis=1)
        undiagnosed_data['risk_category'] = undiagnosed_data['risk_score'].apply(categorise_risk)
        undiagnosed_data['risk_category'] = undiagnosed_data.apply(potentially_undiagnosed, axis=1)

    diagnosed_data = combined_data[combined_data['asthma_diagnosed'] == 1].copy()
    combined_data = pd.concat([diagnosed_data, undiagnosed_data]).sort_values("patient_id")

    # adding n/a values for people who are diagnosed with asthma
    if 'risk_score' not in combined_data.columns:
        combined_data['risk_score'] = pd.NA
    if 'risk_category' not in combined_data.columns:
        combined_data['risk_category'] = pd.NA

    # this will filter the data based on the selected risk group in the gui
    if risk_group == "Diagnosed":
        combined_data = combined_data[combined_data['asthma_diagnosed'] == 1]
    elif risk_group == "Potentially Undiagnosed Asthma":
        combined_data = combined_data[combined_data['risk_category'] == 'Potentially Undiagnosed Asthma']
    elif risk_group != "All":
        combined_data = combined_data[combined_data['risk_category'] == risk_group]

    # Create a prediction report
    prediction_report = []
    # loops through data patient by patient
    for _, patient in combined_data.iterrows():
        if patient['asthma_diagnosed'] == 1:
            prediction_report.append(f"Patient {patient['patient_id']} is diagnosed with asthma.")
        else:
            risk_category = patient['risk_category']
            # checks patient matches filter, and then adds them to report
            if risk_group == "All" or risk_category == risk_group:
                prediction_report.append(f"Patient {patient['patient_id']} has a {risk_category}.")

                # adds the risk factor for the patient to the report
                risk_factors = []
                if patient['bmi'] > 25:
                    risk_factors.append('overweight/obese')
                if patient['smoker'] == 1:
                    risk_factors.append('smoker')
                if patient['air_quality'] in ['Poor', 'Very Poor']:
                    risk_factors.append('poor air quality')
                if patient['asthma_within_family'] == 1:
                    risk_factors.append('asthma within family')
                if patient['temperature'] < 15:
                    risk_factors.append('cold temperature')
                risk_factors.append(f'living in {patient["living_conditions"]} area')

                prediction_report.append(f"Risk factors: {', '.join(risk_factors)}.")

    return prediction_report


def open_files():
    global file_paths
    file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
    if file_paths:
        refresh_data()


def refresh_data():
    if file_paths:
        # get Filters from the ui
        filters = {
            'age_min': age_min_var.get(),
            'age_max': age_max_var.get(),
            'gender': gender_filter_var.get(),
            'location': location_filter_var.get()
        }
        risk_group = risk_filter_var.get()
        prediction_report = generate_prediction_report(filters, risk_group)
        display_report(prediction_report)


# clears old data and shows new report
def display_report(prediction_report):
    report_text.delete(1.0, tk.END)
    for line in prediction_report:
        report_text.insert(tk.END, line + "\n")


def save_report():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'w') as file:
            file.write(report_text.get(1.0, tk.END))


def select_all_locations():
    location_filter_var.set("All")


def select_all_genders():
    gender_filter_var.set("All")


locations = ['Newcastle', 'Gateshead', 'Sunderland', 'Durham', 'Hartlepool', 'Middlesbrough', 'Stockton-on-Tees', 'Darlington']

# GUI Setup
window = tk.Tk()
window.title("Asthma Prediction")
window.geometry("800x600")


filter_frame = tk.Frame(window)
filter_frame.pack(pady=10)


button_frame = tk.Frame(window)
button_frame.pack(pady=10)
tk.Button(button_frame, text="Import CSV Files", command=open_files, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=10)


tk.Label(filter_frame, text="Age Range:").grid(row=0, column=0, padx=5, pady=5)
age_min_var = tk.IntVar(value=0)
age_max_var = tk.IntVar(value=100)
tk.Entry(filter_frame, textvariable=age_min_var, width=5).grid(row=0, column=1, padx=5)
tk.Label(filter_frame, text="to").grid(row=0, column=2, padx=5)
tk.Entry(filter_frame, textvariable=age_max_var, width=5).grid(row=0, column=3, padx=5)


tk.Label(filter_frame, text="Gender:").grid(row=1, column=0, padx=5, pady=5)
gender_filter_var = tk.StringVar(value="All")
gender_combobox = ttk.Combobox(filter_frame, textvariable=gender_filter_var, values=["All", "Male", "Female"])
gender_combobox.grid(row=1, column=1, columnspan=3, padx=5, pady=5)
tk.Button(filter_frame, text="Select All", command=select_all_genders, bg="#4CAF50", fg="white").grid(row=1, column=4, padx=5, pady=5)


tk.Label(filter_frame, text="Location:").grid(row=2, column=0, padx=5, pady=5)
location_filter_var = tk.StringVar(value=locations[0])
location_combobox = ttk.Combobox(filter_frame, textvariable=location_filter_var, values=locations)
location_combobox.grid(row=2, column=1, columnspan=3, padx=5, pady=5)
tk.Button(filter_frame, text="Select All", command=select_all_locations, bg="#4CAF50", fg="white").grid(row=2, column=4, padx=5, pady=5)

tk.Label(filter_frame, text="Risk Group:").grid(row=3, column=0, padx=5, pady=5)
risk_filter_var = tk.StringVar(value="All")
ttk.Combobox(filter_frame, textvariable=risk_filter_var, values=["All", "Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Very High Risk", "Diagnosed", "Potentially Undiagnosed Asthma"]).grid(row=3, column=1, columnspan=3, padx=5, pady=5)
report_text = Text(window, height=20, width=100)
report_text.pack(pady=10)

tk.Button(button_frame, text="Refresh Data", command=refresh_data, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=10)
tk.Button(window, text="Save Report", command=save_report, bg="#4CAF50", fg="white").pack(pady=10)

window.mainloop()

