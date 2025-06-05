"""
1. Track Characteristics
Type: Permanent racing facility with street-like features

Length: ~5.45 km

Laps: ~56

Speed Profile: Mix of high-speed straights and technical corners (notably the long Turn 1 and long back straight)

Grip Level: Medium grip, but rubber buildup improves it through the weekend

Overtaking: Moderate : long back straight with heavy braking into Turn 14 provides good overtaking opportunity

2. Weather Conditions
Unpredictable: Rain is often a factor, especially in spring

Analysis: Historical data shows a mix of dry and wet races; teams need to plan for changing conditions

3. Tire Strategy
Tire Wear: Moderate to high : especially due to long corners and heavy traction zones

Pit Stops: Typically 1 to 2 stops depending on weather and tire degradation

Tire Choices: Medium and hard compounds preferred; softs degrade quickly here

4. Safety Cars
Probability: Medium : accidents in wet conditions or technical failures can trigger safety cars

Strategic Impact: Safety car timing can significantly impact pit strategy; teams often prepare alternate plans

5. Car Setup Preferences
Downforce: Medium : balance needed between straight-line speed and cornering grip

Braking: Multiple heavy braking zones (Turns 6, 11, 14) : critical for brake management and energy recovery systems

DRS Zones: Usually 2 : including a long back straight that is key for overtaking
"""

import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

fastf1.Cache.enable_cache('f1_cache')

# Helper to convert timedelta to seconds
def td_to_sec(td):
    return td.total_seconds() if pd.notnull(td) else np.nan

def get_session_data(year, grand_prix, session_type):
    """Loads session data and handles potential errors, returning None if unsuccessful."""
    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load(telemetry=False, weather=False, messages=False) 
        
        if session.laps.empty:
            print(f"Warning: No lap data found for {year} {grand_prix} {session_type}. Skipping.")
            return None
        return session
    except Exception as e:
        print(f"Error loading {year} {grand_prix} {session_type} data: {e}")
        return None

# Define the Grand Prix for which we are making predictions
GRAND_PRIX_NAME = 'China'

## 1. Data Collection for Training (2024 Race and 2024 Qualifying)

# --- 2024 Race Data for Driver Profiles ---
print(f"Loading 2024 {GRAND_PRIX_NAME} Race data for driver profiles...")
race_2024_session = get_session_data(2024, GRAND_PRIX_NAME, 'R')

if race_2024_session is None:
    print(f"Crucial training data (2024 {GRAND_PRIX_NAME} Race) not available. Exiting.")
    exit()

required_race_cols = ['Driver', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'Compound', 'TyreLife']
missing_race_cols = [col for col in required_race_cols if col not in race_2024_session.laps.columns]
if missing_race_cols:
    print(f"Error: Missing columns in 2024 Race data: {missing_race_cols}. Exiting.")
    exit()

laps_2024_race = race_2024_session.laps[required_race_cols].dropna(subset=['LapTime'])
if laps_2024_race.empty:
    print(f"No valid lap data after dropping NaNs for 2024 {GRAND_PRIX_NAME} Race. Exiting.")
    exit()

laps_2024_race['LapTime_s'] = laps_2024_race['LapTime'].apply(td_to_sec)
laps_2024_race['S1_s'] = laps_2024_race['Sector1Time'].apply(td_to_sec)
laps_2024_race['S2_s'] = laps_2024_race['Sector2Time'].apply(td_to_sec)
laps_2024_race['S3_s'] = laps_2024_race['Sector3Time'].apply(td_to_sec)

# Create driver profiles from 2024 race data
driver_profiles = laps_2024_race.groupby('Driver').agg({
    'S1_s': ['mean', 'min', 'std'],
    'S2_s': ['mean', 'min', 'std'],
    'S3_s': ['mean', 'min', 'std'],
    'LapTime_s': ['mean', 'min', 'std'],
    'Driver': 'count', # Count of laps completed in the race
    'TyreLife': ['mean', 'std'] # Avg and std tyre life in races
})
driver_profiles.columns = ['_'.join(col).strip() for col in driver_profiles.columns.values]
driver_profiles = driver_profiles.rename(columns={'Driver_count': 'laps_completed'})
driver_profiles = driver_profiles.reset_index()

# Debugging: Check how many drivers are in driver_profiles
print(f"Number of drivers in 2024 Race driver_profiles: {len(driver_profiles['Driver'].unique())}")


# --- 2024 Qualifying Data for Training Target (Manual Fastest Lap Calculation) ---
print(f"Loading 2024 {GRAND_PRIX_NAME} Qualifying data for training target...")
qual_2024_session = get_session_data(2024, GRAND_PRIX_NAME, 'Q')

if qual_2024_session == None:
    print(f"Crucial training data (2024 {GRAND_PRIX_NAME} Qualifying) not available. Exiting.")
    exit()

# Get all raw laps from 2024 Qualifying to manually find fastest laps
raw_laps_2024_qual = qual_2024_session.laps

required_qual_cols_all = ['Driver', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'Compound', 'TyreLife']
missing_qual_cols_all = [col for col in required_qual_cols_all if col not in raw_laps_2024_qual.columns]

if missing_qual_cols_all:
    print(f"Error: Missing columns in raw 2024 Qualifying laps: {missing_qual_cols_all}. Exiting.")
    exit()

laps_2024_qual_processed = raw_laps_2024_qual.copy()
laps_2024_qual_processed['S1_s'] = laps_2024_qual_processed['Sector1Time'].apply(td_to_sec)
laps_2024_qual_processed['S2_s'] = laps_2024_qual_processed['Sector2Time'].apply(td_to_sec)
laps_2024_qual_processed['S3_s'] = laps_2024_qual_processed['Sector3Time'].apply(td_to_sec)
laps_2024_qual_processed['LapTime_s'] = laps_2024_qual_processed['LapTime'].apply(td_to_sec)

# Drop rows where critical data for training target is missing
laps_2024_qual_processed = laps_2024_qual_processed.dropna(subset=['S1_s', 'S2_s', 'S3_s', 'LapTime_s'])

if laps_2024_qual_processed.empty:
    print(f"No valid lap data after processing and dropping NaNs for 2024 {GRAND_PRIX_NAME} Qualifying. Exiting.")
    exit()

# Manually get the fastest lap for each driver based on 'LapTime_s'
laps_2024_qual = laps_2024_qual_processed.loc[laps_2024_qual_processed.groupby('Driver')['LapTime_s'].idxmin()]

# Extract sector times and compound for the 2024 qualifying for later use in 2025 prediction merge
# This effectively creates 'sector_times_2024' as implied in your prompt
laps_2024_qual_sectors_only = laps_2024_qual[['Driver', 'S1_s', 'S2_s', 'S3_s', 'Compound', 'TyreLife', 'LapTime_s']].copy() # Keep LapTime_s for speed factor


# Debugging: Check how many drivers are in laps_2024_qual after manual processing
print(f"Number of drivers in laps_2024_qual (manual fastest lap processed): {len(laps_2024_qual['Driver'].unique())}")
print(f"Drivers in laps_2024_qual (manual fastest lap processed): {laps_2024_qual['Driver'].unique().tolist()}")


# Merge driver profiles (from 2024 race) with 2024 qualifying best laps
train_df = pd.merge(laps_2024_qual, driver_profiles, on='Driver', how='left')
# Fill NaN values introduced by the merge (for drivers in qual not in race profiles)
numerical_driver_profile_cols = [col for col in driver_profiles.columns if col != 'Driver']
train_df[numerical_driver_profile_cols] = train_df[numerical_driver_profile_cols].fillna(0)


# Debugging: Check how many drivers are in train_df
print(f"Number of drivers in train_df (after merge and fillna): {len(train_df['Driver'].unique())}")
print(f"train_df head:\n{train_df.head()}")


## 2. Feature Engineering

# One-hot encode 'Compound' for training data
train_df = pd.get_dummies(train_df, columns=['Compound'], prefix='Tyre', dtype=int)

# --- NEW: Add a placeholder for 'TargetLapTime_s' in training data ---
# This feature will be 0 for training data since we don't have a specific 2024 target.
# It will be filled with the custom 2025 target during prediction.
train_df['TargetLapTime_s'] = 0.0 

feature_cols = [
    'S1_s', 'S2_s', 'S3_s', # Best sector times from 2024 qualifying (manual fastest)
    'S1_s_mean', 'S1_s_min', 'S1_s_std', # Driver's race sector stats (from 2024 race)
    'S2_s_mean', 'S2_s_min', 'S2_s_std',
    'S3_s_mean', 'S3_s_min', 'S3_s_std',
    'LapTime_s_mean', 'LapTime_s_min', 'LapTime_s_std', # Driver's race lap time stats (from 2024 race)
    'laps_completed', # Number of race laps completed by driver
    'TyreLife_mean', 'TyreLife_std', # Tyre life stats from 2024 race
    'TargetLapTime_s' # The new feature to guide prediction
]

# Get all unique tyre columns created during training to ensure consistency
all_tyre_cols_in_training = [col for col in train_df.columns if col.startswith('Tyre_')]
feature_cols.extend(all_tyre_cols_in_training)

# Ensure feature_cols only contains columns present in train_df
feature_cols = [col for col in feature_cols if col in train_df.columns]

X_train = train_df[feature_cols]
y_train = train_df['LapTime_s']

# Debugging: Check final shape of X_train
print(f"Final X_train shape before scaling: {X_train.shape}")


# Standardize features
scaler = StandardScaler()

## 3. Model Selection and Validation: K-fold Cross-Validation
print("\n--- Performing K-Fold Cross-Validation ---")

if X_train.shape[0] < 2: # Need at least 2 samples for k-fold with n_splits=2 or more
    print(f"Warning: Not enough samples for K-Fold Cross-Validation (n_samples={X_train.shape[0]}). Skipping cross-validation and training directly.")
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    final_model = GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42)
    final_model.fit(X_train_scaled, y_train)
else:
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    
    n_splits_kf = min(5, X_train_scaled.shape[0])
    kf = KFold(n_splits=n_splits_kf, shuffle=True, random_state=42)
    fold_mae_scores = []
    fold_rmse_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_scaled)):
        X_train_fold, X_val_fold = X_train_scaled.iloc[train_index], X_train_scaled.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model = GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42)
        model.fit(X_train_fold, y_train_fold)
        val_preds = model.predict(X_val_fold)

        mae = mean_absolute_error(y_val_fold, val_preds)
        rmse = np.sqrt(mean_squared_error(y_val_fold, val_preds))
        fold_mae_scores.append(mae)
        fold_rmse_scores.append(rmse)
        print(f"Fold {fold+1}: MAE = {mae:.3f}s, RMSE = {rmse:.3f}s")

    print(f"\nAverage MAE across folds: {np.mean(fold_mae_scores):.3f}s")
    print(f"Average RMSE across folds: {np.mean(fold_rmse_scores):.3f}s")
    print("--- Cross-Validation Complete ---")

    final_model = GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42)
    final_model.fit(X_train_scaled, y_train)


## Prepare 2025 qualifying features for prediction (Using your custom data)
print(f"\n--- Preparing 2025 {GRAND_PRIX_NAME} Qualifying data for prediction (using custom input) ---")

# YOUR CUSTOM 2025 QUALIFYING DATA
qualifying_2025 = pd.DataFrame({
    "Driver": ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
               "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
               "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.",
               "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"],
    "QualifyingTime (s)": [90.641, 90.723, 90.793, 90.817, 90.927,
                           91.021, 91.079, 91.103, 91.638, 91.706,
                           91.625, 91.632, 91.688, 91.773, 91.840,
                           91.992, 92.018, 92.092, 92.141, 92.174]
})

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)


# Prepare the 2025 prediction input
qual_2025_prediction_input = qualifying_2025[['DriverCode', 'QualifyingTime (s)']].copy()

# Merge with 2024 qualifying data to bring in Compound and TyreLife, but not sectors this time
# We'll calculate new sectors based on the custom QualifyingTime (s)
qual_2025_prediction_input = qual_2025_prediction_input.merge(
    laps_2024_qual_sectors_only[['Driver', 'Compound', 'TyreLife']],
    left_on='DriverCode',
    right_on='Driver',
    how='left'
)
# Drop the duplicate 'Driver' column from the merge, keep 'DriverCode' for now
qual_2025_prediction_input.drop(columns='Driver', inplace=True)


# Calculate average sector ratios from 2024 overall qualifying data
# These ratios will be used to proportionally distribute the custom 2025 target time into sectors
avg_s1_ratio = (laps_2024_qual_sectors_only['S1_s'] / laps_2024_qual_sectors_only['LapTime_s']).mean()
avg_s2_ratio = (laps_2024_qual_sectors_only['S2_s'] / laps_2024_qual_sectors_only['LapTime_s']).mean()
avg_s3_ratio = (laps_2024_qual_sectors_only['S3_s'] / laps_2024_qual_sectors_only['LapTime_s']).mean()

# Apply these average ratios to the custom 2025 QualifyingTime (s) for ALL drivers
qual_2025_prediction_input['S1_s'] = qual_2025_prediction_input['QualifyingTime (s)'] * avg_s1_ratio
qual_2025_prediction_input['S2_s'] = qual_2025_prediction_input['QualifyingTime (s)'] * avg_s2_ratio
qual_2025_prediction_input['S3_s'] = qual_2025_prediction_input['QualifyingTime (s)'] * avg_s3_ratio


# Assign default Compound and TyreLife if missing (e.g., for drivers not in 2024 quali data)
most_common_compound_2024 = laps_2024_qual_sectors_only['Compound'].mode()[0] if not laps_2024_qual_sectors_only['Compound'].empty else 'SOFT'
qual_2025_prediction_input['Compound'] = qual_2025_prediction_input['Compound'].fillna(most_common_compound_2024)
qual_2025_prediction_input['TyreLife'] = qual_2025_prediction_input['TyreLife'].fillna(2.0) # Assume 2.0 as a typical quali tyre life


# Now select the features for prediction. Rename 'DriverCode' to 'Driver'.
qual_2025_prediction_input.rename(columns={'DriverCode': 'Driver'}, inplace=True)
qual_2025_prediction_input = qual_2025_prediction_input[[
    'Driver', 'S1_s', 'S2_s', 'S3_s', 'Compound', 'TyreLife', 'QualifyingTime (s)' 
]].rename(columns={
    'QualifyingTime (s)': 'TargetLapTime_s' # Rename for consistency with feature_cols
})


# Debugging: Check qual_2025_prediction_input after custom data integration
print(f"Number of drivers in qual_2025_prediction_input (from custom data): {len(qual_2025_prediction_input['Driver'].unique())}")
print(f"Drivers in qual_2025_prediction_input (from custom data): {qual_2025_prediction_input['Driver'].unique().tolist()}")
print(f"qual_2025_prediction_input head:\n{qual_2025_prediction_input.head()}")


# Merge with driver profiles (from 2024 race data)
test_df = pd.merge(qual_2025_prediction_input, driver_profiles, on='Driver', how='left')

# Fill NaN values for driver profile features (those from 2024 race data)
profile_cols_to_fill = [col for col in numerical_driver_profile_cols if col in test_df.columns]
test_df[profile_cols_to_fill] = test_df[profile_cols_to_fill].fillna(0) # Fill numerical NaNs with 0

# Debugging: Check test_df after merge
print(f"Number of drivers in test_df (after merge with driver_profiles and fillna): {len(test_df['Driver'].unique())}")
print(f"Drivers in test_df: {test_df['Driver'].unique().tolist()}")
print(f"test_df head:\n{test_df.head()}")


# One-hot encode 'Compound' for test data
test_df = pd.get_dummies(test_df, columns=['Compound'], prefix='Tyre', dtype=int)

# --- CRITICAL FIX: Align test_df columns with X_train columns ---
# This ensures that test_df has all the columns from X_train, and in the same order
test_df_aligned = pd.DataFrame(0.0, index=test_df.index, columns=feature_cols) # Initialize with zeros
for col in test_df.columns:
    if col in test_df_aligned.columns:
        test_df_aligned[col] = test_df[col]
# No need to fillna again as it was initialized with 0.0

# Ensure the order of columns is the same as in X_train
X_test = test_df_aligned[feature_cols]

# Debugging: Check final X_test shape
print(f"Final X_test shape before scaling: {X_test.shape}")


# Scale X_test using the *same* scaler fitted on X_train
if X_test.empty:
    print("No valid data for 2025 qualifying prediction after feature engineering. Exiting.")
else:
    X_test_scaled = scaler.transform(X_test)

    # Predict qualifying lap times for 2025
    predicted_lap_times = final_model.predict(X_test_scaled)
    
    # Ensure the 'Driver' column comes from the input that guarantees all 2025 drivers
    grid_prediction = pd.DataFrame({
        'Driver': qual_2025_prediction_input['Driver'],
        'PredictedLapTime': predicted_lap_times
    })

    # Sort by predicted lap time ascending -> grid order
    grid_prediction = grid_prediction.sort_values('PredictedLapTime').reset_index(drop=True)

    print(f"\nTop 10 predicted grid positions for 2025 {GRAND_PRIX_NAME} GP qualifying:")
    print(grid_prediction[['Driver', 'PredictedLapTime']].head(10))
