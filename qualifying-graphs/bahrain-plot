import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache("f1_cache")

# Load 2024 Bahrain race data
session_2024 = fastf1.get_session(2024, "Bahrain", "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Average sector times per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# 2025 Bahrain GP qualifying data
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"],
    "QualifyingTime (s)": [90.423, 90.267, 89.841, 90.175, 90.009, 90.772, 90.216, 91.886, 91.303, 90.680, 92.067, 91.886, 92.283]
})

# Wet performance factor mapping
driver_wet_performance = {
    "VER": 0.975196, "HAM": 0.976464, "LEC": 0.975862, "NOR": 0.978179, "ALO": 0.972655, 
    "RUS": 0.968678, "SAI": 0.978754, "TSU": 0.996338, "OCO": 0.981810, "GAS": 0.978832, "STR": 0.979857
}
qualifying_2025["WetPerformanceFactor"] = qualifying_2025["Driver"].map(driver_wet_performance)

# Season points mapping
season_points = {
    "VER": 61, "NOR": 62, "PIA": 45, "LEC": 20, "RUS": 45,
    "HAM": 15, "GAS": 0, "ALO": 0, "TSU": 3, "SAI": 1, "HUL": 2, "OCO": 10, "STR": 1
}
qualifying_2025["SeasonPoints"] = qualifying_2025["Driver"].map(season_points)

# Merge qualifying with sector times
merged_data = qualifying_2025.merge(sector_times_2024, on="Driver", how="left")

# Prepare features
X = merged_data[[
    "QualifyingTime (s)", 
    "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", 
    "WetPerformanceFactor", 
    "SeasonPoints"
]].fillna(0)

# Amplify importance of QualifyingTime and SeasonPoints
X["QualifyingTime (s)"] *= 2.5  
X["SeasonPoints"] *= 2.5        

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Prepare target variable (average lap time from 2024 race)
y = merged_data.merge(
    laps_2024.groupby("Driver")["LapTime (s)"].mean(),
    left_on="Driver",
    right_index=True
)["LapTime (s)"]

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=38)

# Train model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict 2025 race times on full dataset
predicted_race_times = model.predict(X_scaled)
merged_data["PredictedRaceTime (s)"] = predicted_race_times
merged_data = merged_data.sort_values(by="PredictedRaceTime (s)")

print("\n🏁 Predicted 2025 Bahrain GP Winner 🏁\n")
print(merged_data[["Driver", "PredictedRaceTime (s)"]])

# Evaluate model error on test set
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Plot feature importances
feature_importance = model.feature_importances_
features = X_scaled.columns

plt.figure(figsize=(8,5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.show()
