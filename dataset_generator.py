import pandas as pd
import random
import tkinter as tk
from tkinter import filedialog
from faker import Faker

fake = Faker()


locations = ['Newcastle', 'Gateshead', 'Sunderland', 'Durham', 'Hartlepool', 'Middlesbrough', 'Stockton-on-Tees',
             'Darlington']
air_quality_levels = ['Good', 'Moderate', 'Poor', 'Very Poor']
living_conditions = ['Urban', 'Suburban', 'Rural']


def generate_dataset(num_samples, selected_locations, file_path):
    data = []
    for i in range(num_samples):

        patient_id = i + 1
        age = random.randint(0, 100)
        gender = random.choice(['Male', 'Female'])
        air_quality = random.choice(air_quality_levels)
        living_condition = random.choice(living_conditions)
        location = random.choice(selected_locations)
        temperature = round(random.uniform(10, 30), 1)
        humidity = random.randint(30, 90)
        smoker = random.choices([0, 1], weights=[random.uniform(85, 90), random.uniform(10, 15)])[0]
        asthma_within_family = random.choice([0, 1])
        bmi = round(random.uniform(15, 45), 1)

        if gender == 'Male':
            fev1 = round(random.uniform(3.5, 4.5), 1)
        else:
            fev1 = round(random.uniform(2.5, 3.25), 1)

        if bmi > 25:
            fev1 -= 0.75
        if living_condition == 'Urban':
            fev1 -= 0.25
        if smoker == 1:
            fev1 -= 0.5
        if air_quality in ['Poor', 'Very Poor']:
            fev1 -= 0.25

        asthma_diagnosed = random.choices([0, 1], weights=[90, round(random.uniform(5,13),1)])[0]  # 5-13% chance of being diagnosed with asthma

        data.append([patient_id, age, gender, air_quality, living_condition, location, temperature, humidity,
                     smoker, asthma_within_family, fev1, bmi, asthma_diagnosed])

    data_frame = pd.DataFrame(data, columns=['patient_id', 'age', 'gender', 'air_quality', 'living_conditions',
                                     'location', 'temperature', 'humidity', 'smoker', 'asthma_within_family', 'fev1',
                                     'bmi', 'asthma_diagnosed'])
    data_frame.to_csv(file_path, index=False)


def save_dataset():

    num_patients = int(num_patients_entry.get())

    # Get the names of selected locations from location_vars
    selected_locations = [location for location, var in location_vars.items() if var.get()]

    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])

    generate_dataset(num_patients, selected_locations, file_path)

    window.destroy()


window = tk.Tk()
window.title("Dataset Generator")

tk.Label(window, text="Number of Patients:").pack()
num_patients_entry = tk.Entry(window)
num_patients_entry.pack()

tk.Label(window, text="Select Locations:").pack()
location_vars = {location: tk.BooleanVar(value=True) for location in locations}
for location, var in location_vars.items():
    tk.Checkbutton(window, text=location, variable=var).pack(anchor='w')

tk.Button(window, text="Generate and Save Dataset", command=save_dataset).pack()

window.mainloop()

