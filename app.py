from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model features
with open("model_features.json", "r") as f:
    features = json.load(f)

# Load models
targets = ['Heat_Stress_Level', 'Heat_Stress_Severity', 'Body_Temperature_C', 'Respiration_Rate_bpm', 'Cooling_Effect']
models = {t: joblib.load(f"{t}_model.pkl") for t in targets}

# Load label encoders
with open("label_encoders.json", "r") as f:
    label_classes = json.load(f)

label_encoders = {}
for col, classes in label_classes.items():
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    label_encoders[col] = le

def calculate_stress_level(env):
    temp = env['temperature']
    humidity = env['humidity']
    wind = env['wind_speed']
    solar = env['solar_radiation']
    stress_index = temp * 0.5 + humidity * 0.3 + solar * 0.15 - wind * 2
    if stress_index < 50:
        return 0  # Normal
    elif stress_index < 65:
        return 1  # Mild
    elif stress_index < 80:
        return 2  # Moderate
    else:
        return 3  # Severe

def get_advice(level):
    advice_bank = {
        0: "No immediate action needed. Ensure animals have shade and water.",
        1: "Monitor animals closely and provide extra water access.",
        2: "Reduce handling. Cool housing and ensure airflow.",
        3: "Take immediate action: misting, fans, or relocate animals to cooler areas."
    }
    return advice_bank.get(level, "Ensure safety protocols are in place.")

@app.route('/', methods=['GET', 'POST'])
def index():
    species_list = ['cattle', 'goat', 'sheep']
    breed_options = {sp: list(label_encoders[f"{sp.capitalize()}_Breed"].classes_) for sp in species_list}
    age_options = {sp: list(label_encoders[f"{sp}_age_group"].classes_) for sp in species_list}
    if request.method == 'POST':
        species = request.form['species']
        breed = request.form['breed']
        age_group = request.form['age_group']
        try:
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            wind_speed = float(request.form['wind_speed'])
            solar_radiation = float(request.form['solar_radiation'])
        except ValueError:
            error = "Please enter valid numerical values for environmental conditions."
            return render_template('index.html', species_list=species_list, breed_options=breed_options, age_options=age_options, error=error)

        # Validate input ranges
        if not (20 <= temperature <= 45):
            error = "Temperature should be between 20°C and 45°C."
            return render_template('index.html', species_list=species_list, breed_options=breed_options, age_options=age_options, error=error)
        if not (10 <= humidity <= 100):
            error = "Humidity should be between 10% and 100%."
            return render_template('index.html', species_list=species_list, breed_options=breed_options, age_options=age_options, error=error)
        if not (0 <= wind_speed <= 10):
            error = "Wind Speed should be between 0 and 10 m/s."
            return render_template('index.html', species_list=species_list, breed_options=breed_options, age_options=age_options, error=error)
        if not (0 <= solar_radiation <= 1200):
            error = "Solar Radiation should be between 0 and 1200 W/m²."
            return render_template('index.html', species_list=species_list, breed_options=breed_options, age_options=age_options, error=error)

        env = {
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'solar_radiation': solar_radiation
        }

        full_input = {
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'solar_radiation': solar_radiation,
            'Cattle_Breed': 'Unknown',
            'Goat_Breed': 'Unknown',
            'Sheep_Breed': 'Unknown',
            'cattle_age_group': 'Unknown',
            'goat_age_group': 'Unknown',
            'sheep_age_group': 'Unknown'
        }

        breed_col = f"{species.capitalize()}_Breed"
        age_col = f"{species}_age_group"
        full_input[breed_col] = breed
        full_input[age_col] = age_group

        for col in label_encoders:
            val = full_input[col]
            le = label_encoders[col]
            if val in le.classes_:
                full_input[col] = le.transform([val])[0]
            else:
                full_input[col] = None

        full_input = {k: v for k, v in full_input.items() if v is not None}
        for f in features:
            if f not in full_input:
                full_input[f] = 0.0

        X = pd.DataFrame([full_input])[features]

        predictions = {}
        for target in ['Body_Temperature_C', 'Respiration_Rate_bpm', 'Cooling_Effect']:
            predictions[target] = models[target].predict(X)[0]

        predictions['Heat_Stress_Level'] = calculate_stress_level(env)
        predictions['advice'] = get_advice(predictions['Heat_Stress_Level'])

        return render_template('result.html', species=species, breed=breed, age_group=age_group, env=env, predictions=predictions)

    return render_template('index.html', species_list=species_list, breed_options=breed_options, age_options=age_options)

if __name__ == '__main__':
    app.run(debug=True)
