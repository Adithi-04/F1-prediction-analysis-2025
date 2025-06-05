import fastf1
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# ========== 1. LOAD HISTORICAL DATA ==========
# Load 2024 Miami GP race session
session_2024 = fastf1.get_session(2024, "Miami", "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)


# Convert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Group by driver to get average sector times per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# ========== 2. 2025 QUALIFYING DATA (hypothetical) ==========
qualifying_2025 = pd.DataFrame({
    "Driver": ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
               "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
               "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.",
               "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"],
    "QualifyingTime (s)": [86.375, 86.385, 86.269, 86.204, 87.006,
                           86.754, 86.987, 86.271, 86.943, 86.682,
                           86.824, 87.473, 87.604, 87.830, 86.569,
                           87.710, 87.999, 87.186, 87.151, 87.363]
})

# Add constructor (team) information
constructor_mapping = {
    "Oscar Piastri": "McLaren", "George Russell": "Mercedes", "Lando Norris": "McLaren", "Max Verstappen": "Red Bull",
    "Lewis Hamilton": "Ferrari", "Charles Leclerc": "Ferrari", "Isack Hadjar": "Racing Bulls Honda RBPT", 
    "Andrea Kimi Antonelli": "Mercedes", "Yuki Tsunoda": "Red Bull", "Alexander Albon": "Williams",
    "Esteban Ocon": "Haas", "Nico Hülkenberg": "Kick Sauber", "Fernando Alonso": "Aston Martin", 
    "Lance Stroll": "Aston Martin", "Carlos Sainz Jr.": "Williams", "Pierre Gasly": "Alpine",
    "Oliver Bearman": "Haas", "Jack Doohan": "Alpine", "Gabriel Bortoleto": "Kick Sauber", "Liam Lawson": "Racing Bulls Honda RBPT"
}
qualifying_2025["Constructor"] = qualifying_2025["Driver"].map(constructor_mapping)

# Define constructor points (hypothetical)
constructor_points = {
    "Red Bull": 92, "Mercedes": 118, "Ferrari": 84, "Aston Martin": 14, "Alpine": 7,
    "McLaren": 203, "Haas": 20, "Racing Bulls Honda RBPT": 8, "Williams": 25, "Kick Sauber": 6
}
qualifying_2025["ConstructorPoints"] = qualifying_2025["Constructor"].map(constructor_points)

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}
qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# ========== 3. FEATURE ENGINEERING ==========
# Merge qualifying data with sector times
merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode", right_on="Driver", how="left")

# Feature: Sector variance (consistency)
merged_data["SectorVariance"] = merged_data[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].var(axis=1)

# Example pit stop data (could be improved with real stats)
pit_data = {
    "Constructor": ["Red Bull", "Mercedes", "Ferrari", "Aston Martin", "Alpine", "McLaren", "Haas", "Racing Bulls Honda RBPT", "Williams", "Kick Sauber"],
    "AvgPitTime": [2.3, 2.5, 2.4, 2.6, 2.7, 2.35, 2.55, 2.65, 2.6, 2.7]
}
pit_df = pd.DataFrame(pit_data)
merged_data = merged_data.merge(pit_df, on="Constructor", how="left")

# ========== 4. PREPARE TRAINING DATA ==========
# Target: average race lap time per driver
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()["LapTime (s)"]

# Features for model
feature_cols = [
    "QualifyingTime (s)",
    "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",
    "SectorVariance",
    "ConstructorPoints",
    "AvgPitTime"
]
X = merged_data[feature_cols].fillna(0)

# ========== 5. MODEL STACKING & TRAINING ==========
# Define base models
estimators = [
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42)),
    ('lgbm', LGBMRegressor(random_state=42))
]
stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
)

# TimeSeries split (for robustness)
tscv = TimeSeriesSplit(n_splits=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)

# Fit model
stack_model.fit(X_train, y_train)

# ========== 6. PREDICT & RANK ==========
predicted_race_times = stack_model.predict(X)
qualifying_2025["PredictedRaceTime (s)"] = predicted_race_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)").reset_index(drop=True)
qualifying_2025["Position"] = qualifying_2025.index + 1
qualifying_2025["Position"] = qualifying_2025["Position"].apply(
    lambda x: f"{x}{'st' if x == 1 else 'nd' if x == 2 else 'rd' if x == 3 else 'th'}"
)

# ========== 7. MONTE CARLO SIMULATION ==========
# (Optional: to estimate uncertainty)
num_simulations = 1000
sim_results = np.zeros((num_simulations, len(predicted_race_times)))

for i in range(num_simulations):
    noise = np.random.normal(1, 0.01, len(predicted_race_times))
    sim_times = predicted_race_times * noise
    sim_results[i, :] = np.argsort(np.argsort(sim_times)) + 1  # 1 = best

# Calculate probabilities of podium finish
podium_probs = (sim_results <= 3).mean(axis=0)
qualifying_2025["PodiumChance (%)"] = (podium_probs * 100).round(1)

# ========== 8. OUTPUT & EVALUATION ==========
print("\n🏁 Predicted 2025 Miami GP Results (Stacked Model, Monte Carlo) 🏁\n")
print(qualifying_2025[["Position", "Driver", "Constructor", "PredictedRaceTime (s)", "ConstructorPoints", "PodiumChance (%)"]])

# Evaluate model
y_pred = stack_model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# ========== 9. (Optional) EXPORT ==========
# qualifying_2025.to_csv("predicted_2025_miami_gp.csv", index=False)
