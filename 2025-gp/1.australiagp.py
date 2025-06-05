"""
 Australian GP: Key Analytical Points
1. Track Characteristics
Type: Semi-street circuit
Length: ~5.3 km 
Laps: ~58
Speed Profile: Medium to high-speed corners with a few long straights
Grip Level: Often low at the start of the weekend (due to being a street circuit)
Overtaking: Historically difficult but improved with layout changes in 2022

2. Weather Conditions
Unpredictable:
Analysis: Check historical race-day conditions (dry vs. wet).


3.Tire Strategy
Tire Wear: Low to moderate
Pit Stops: Often a one-stop race unless rain or safety cars influence strategy
Tire Choices: Mediums and hards are commonly used

4.Safety Cars
High probability due to walls, tight corners, and first-lap incidents
Strategic Impact: Can create unexpected pit stop windows

5.Car Setup Preferences
Downforce: Medium to high
Braking: Several hard braking zones: important for ERS recovery
DRS Zones: Usually 2 to 3
"""

import numpy as np
import pandas as pd
import fastf1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Set random seeds for reproducibility
np.random.seed(39)
fastf1.Cache.enable_cache("f1_cache")

# 1. DATA LOADING
session_2024 = fastf1.get_session(2024, 3, "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# 2. TRACK FEATURES & QUALIFYING DATA
qualifying_2025 = pd.DataFrame({
    "Driver": ["Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell", 
               "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
               "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"],
    "QualifyingTime (s)": [75.096, 75.180, 75.481, 75.546, 75.670, 75.737,
                           75.755, 75.973, 75.980, 76.062, 76.4, 76.5]
})

track_features = {
    "TrackType": [1]*12,                 # Semi-street circuit
    "TireWear": [1.5]*12,               # Low-moderate  
    "SafetyCarProb": [0.7]*12,          # High SC chance
    "DownforceLevel": [2.5]*12,         # Medium-high
    "WeatherUnpredictability": [0.6]*12 # Variable
}

# 3. FEATURE PREPARATION
driver_mapping = {
    "Lando Norris": "NOR", "Oscar Piastri": "PIA", "Max Verstappen": "VER",
    "George Russell": "RUS", "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB",
    "Charles Leclerc": "LEC", "Lewis Hamilton": "HAM", "Pierre Gasly": "GAS",
    "Carlos Sainz": "SAI", "Fernando Alonso": "ALO", "Lance Stroll": "STR"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)
track_features_df = pd.DataFrame(track_features)

full_features = pd.concat([
    qualifying_2025.reset_index(drop=True), 
    track_features_df.reset_index(drop=True)
], axis=1)

# Merge with 2024 race lap times on DriverCode and Driver
merged_data = full_features.merge(
    laps_2024[["Driver", "LapTime (s)"]],
    left_on="DriverCode",
    right_on="Driver"
).dropna()

# 4. MODEL INPUT
feature_cols = [
    "QualifyingTime (s)", "TrackType", "TireWear", "SafetyCarProb",
    "DownforceLevel", "WeatherUnpredictability"
]

X = merged_data[feature_cols]
y = merged_data["LapTime (s)"]

# 5. PREPROCESSING PIPELINE
preprocessor = ColumnTransformer(
    transformers=[
        ('scale', RobustScaler(), feature_cols)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=39))
])

# 6. TRAIN-TEST SPLIT & MODEL TRAINING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
pipeline.fit(X_train, y_train)

# 7. PREDICTION ON FULL DATA
full_features["PredictedRaceTime (s)"] = pipeline.predict(full_features[feature_cols])
results = full_features.sort_values("PredictedRaceTime (s)")[["Driver", "PredictedRaceTime (s)"]]

# 8. OUTPUT
print("\n🏁 Final 2025 Australian GP Predictions 🏁")
print("=======================================")
print(results.to_string(index=False))

# 9. MODEL EVALUATION
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\n✅ Model MAE: {mae:.3f} seconds")
