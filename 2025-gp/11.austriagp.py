import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 2025 Austria GP data
aus_quali_2025 = pd.DataFrame({
    "Driver": ["NOR", "LEC", "PIA", "HAM", "RUS", "LAW", "VER", "BOR", "ANT", "GAS"],
    "QualifyingTime (s)": [63.971, 64.492, 64.554, 64.582, 64.763, 64.926, 64.929, 65.132, 65.276, 65.649]
})

season_points = {
    "NOR": 201, "LEC": 119, "PIA": 216, "HAM": 91, "RUS": 146,
    "LAW": 12, "VER": 155, "BOR": 4, "ANT": 63, "GAS": 11
}

aus_quali_2025["SeasonPoints"] = aus_quali_2025["Driver"].map(season_points)

print("🤖 ML BEST PRACTICES FOR F1 PREDICTION 🤖\n")

# =============================================
# FEATURE ENGINEERING (RECOMMENDED APPROACH)
# =============================================
print("=" * 60)
print("FEATURE ENGINEERING - THE RECOMMENDED ML APPROACH")
print("=" * 60)

def create_qualifying_focused_features(df):
    """
    Create features that emphasize qualifying performance
    This is the BEST approach for ML because it:
    1. Gives the model multiple ways to understand qualifying importance
    2. Captures non-linear relationships
    3. Provides domain knowledge to the model
    4. Allows the model to learn the relative importance naturally
    """
    
    features = pd.DataFrame()
    
    # 1. RAW QUALIFYING FEATURES (Most Important)
    features['quali_time'] = df['QualifyingTime (s)']
    features['quali_delta_from_pole'] = df['QualifyingTime (s)'] - df['QualifyingTime (s)'].min()
    features['quali_rank'] = df['QualifyingTime (s)'].rank()
    features['quali_percentile'] = df['QualifyingTime (s)'].rank(pct=True)
    
    # 2. NON-LINEAR QUALIFYING FEATURES
    features['quali_delta_squared'] = features['quali_delta_from_pole'] ** 2
    features['quali_delta_log'] = np.log(features['quali_delta_from_pole'] + 0.001)  # +0.001 to avoid log(0)
    
    # 3. QUALIFYING INTERACTION FEATURES
    features['quali_time_normalized'] = (df['QualifyingTime (s)'] - df['QualifyingTime (s)'].mean()) / df['QualifyingTime (s)'].std()
    
    # 4. REDUCED POINTS FEATURES (Secondary importance)
    features['points_normalized'] = (df['SeasonPoints'] - df['SeasonPoints'].min()) / (df['SeasonPoints'].max() - df['SeasonPoints'].min())
    features['points_rank'] = df['SeasonPoints'].rank(ascending=False)
    
    # 5. INTERACTION FEATURES (Qualifying x Points)
    features['quali_points_interaction'] = features['quali_delta_from_pole'] * features['points_normalized']
    
    return features

# Create features
X_features = create_qualifying_focused_features(aus_quali_2025)

print("Generated Features:")
print(X_features.head())
print(f"\nFeature count: {X_features.shape[1]}")
print(f"Features: {list(X_features.columns)}")

# =============================================
# CREATE REALISTIC TRAINING DATA
# =============================================
print("\n" + "=" * 60)
print("CREATING REALISTIC TRAINING DATA")
print("=" * 60)

def create_realistic_training_data(n_samples=200):
    """
    Create synthetic training data that reflects real F1 relationships
    """
    np.random.seed(42)
    
    # Generate realistic qualifying times (63-67 seconds)
    quali_times = np.random.uniform(63.5, 67.0, n_samples)
    
    # Generate realistic season points (0-250)
    season_points = np.random.uniform(0, 250, n_samples)
    
    # Create training dataframe
    train_df = pd.DataFrame({
        'QualifyingTime (s)': quali_times,
        'SeasonPoints': season_points
    })
    
    # Create features
    train_features = create_qualifying_focused_features(train_df)
    
    # Create realistic target (race lap time)
    # This simulates real F1 relationships where:
    # - Qualifying performance is the strongest predictor
    # - Points (experience) provide small advantage
    # - There's some randomness (strategy, luck, etc.)
    
    base_race_time = 71.0
    
    # Strong qualifying impact (1.2x multiplier)
    quali_impact = train_features['quali_delta_from_pole'] * 1.2
    
    # Non-linear qualifying effect (penalizes being far from pole)
    quali_nonlinear = train_features['quali_delta_squared'] * 0.1
    
    # Small points advantage
    points_impact = -train_features['points_normalized'] * 0.2
    
    # Race-day randomness
    randomness = np.random.normal(0, 0.15, n_samples)
    
    # Combine all effects
    race_times = base_race_time + quali_impact + quali_nonlinear + points_impact + randomness
    
    return train_features, race_times

# Create training data
X_train, y_train = create_realistic_training_data(200)

print(f"Training data shape: {X_train.shape}")
print(f"Target variable stats: Mean={y_train.mean():.3f}, Std={y_train.std():.3f}")

# =============================================
# MODEL SELECTION AND HYPERPARAMETER TUNING
# =============================================
print("\n" + "=" * 60)
print("MODEL SELECTION AND HYPERPARAMETER TUNING")
print("=" * 60)

# Test different models
models = {
    'Gradient Boosting (Recommended)': GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    'Gradient Boosting (Tuned)': GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
}

# Cross-validation to select best model
best_model = None
best_score = float('inf')
model_scores = {}

for name, model in models.items():
    # Use cross-validation to get robust performance estimate
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    mean_score = -cv_scores.mean()
    model_scores[name] = mean_score
    
    print(f"{name}: MAE = {mean_score:.4f} (±{cv_scores.std():.4f})")
    
    if mean_score < best_score:
        best_score = mean_score
        best_model = model

print(f"\nBest Model: {list(models.keys())[list(models.values()).index(best_model)]}")

# =============================================
# TRAIN BEST MODEL AND MAKE PREDICTIONS
# =============================================
print("\n" + "=" * 60)
print("TRAINING BEST MODEL AND MAKING PREDICTIONS")
print("=" * 60)

# Train the best model
best_model.fit(X_train, y_train)

# Make predictions
predictions = best_model.predict(X_features)

# Add predictions to dataframe
aus_quali_2025["ML_Predicted_Time"] = predictions
result_df = aus_quali_2025.sort_values("ML_Predicted_Time")

print("🏆 FINAL PREDICTIONS:")
print(f"{'Pos':<4} {'Driver':<6} {'Quali Time':<12} {'Points':<8} {'Predicted Time':<15}")
print("-" * 55)

for i, row in result_df.iterrows():
    pos = list(result_df.index).index(i) + 1
    print(f"{pos:<4} {row['Driver']:<6} {row['QualifyingTime (s)']:<12.3f} {row['SeasonPoints']:<8} {row['ML_Predicted_Time']:<15.3f}")

print(f"\n🎯 WINNER: {result_df.iloc[0]['Driver']}")

# =============================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X_features.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance (Top 8):")
for i, row in feature_importance.head(8).iterrows():
    print(f"{row['Feature']:<25}: {row['Importance']:.4f} ({row['Importance']*100:.1f}%)")

# Calculate qualifying vs non-qualifying importance
quali_features = ['quali_time', 'quali_delta_from_pole', 'quali_rank', 'quali_percentile', 
                  'quali_delta_squared', 'quali_delta_log', 'quali_time_normalized']
quali_importance = feature_importance[feature_importance['Feature'].isin(quali_features)]['Importance'].sum()
total_importance = feature_importance['Importance'].sum()

print(f"\nQualifying-related features: {quali_importance:.3f} ({quali_importance/total_importance*100:.1f}%)")
print(f"Non-qualifying features: {1-quali_importance:.3f} ({(1-quali_importance)/total_importance*100:.1f}%)")

# =============================================
# VISUALIZATION
# =============================================
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Feature Importance
axes[0,0].barh(feature_importance.head(8)['Feature'], feature_importance.head(8)['Importance'])
axes[0,0].set_xlabel('Importance')
axes[0,0].set_title('Top 8 Feature Importance')

# Plot 2: Predicted Results
positions = range(1, len(result_df) + 1)
bars = axes[0,1].barh(result_df['Driver'], result_df['ML_Predicted_Time'], 
                     color=['gold', 'silver', '#CD7F32'] + ['lightblue']*7)
axes[0,1].set_xlabel('Predicted Race Time (s)')
axes[0,1].set_title('ML Predicted Race Times')
axes[0,1].invert_yaxis()

# Plot 3: Qualifying vs Predicted
axes[1,0].scatter(result_df['QualifyingTime (s)'], result_df['ML_Predicted_Time'], 
                 c='red', alpha=0.7, s=100)
for i, driver in enumerate(result_df['Driver']):
    axes[1,0].annotate(driver, 
                      (result_df.iloc[i]['QualifyingTime (s)'], 
                       result_df.iloc[i]['ML_Predicted_Time']), 
                      xytext=(5, 5), textcoords='offset points')
axes[1,0].set_xlabel('Qualifying Time (s)')
axes[1,0].set_ylabel('Predicted Race Time (s)')
axes[1,0].set_title('Qualifying vs Predicted Race Performance')

# Plot 4: Model Comparison
model_names = list(model_scores.keys())
scores = list(model_scores.values())
axes[1,1].bar(range(len(model_names)), scores)
axes[1,1].set_xlabel('Model')
axes[1,1].set_ylabel('Cross-Validation MAE')
axes[1,1].set_title('Model Performance Comparison')
axes[1,1].set_xticks(range(len(model_names)))
axes[1,1].set_xticklabels([name.split('(')[0].strip() for name in model_names], rotation=45)

plt.tight_layout()
plt.show()

# # =============================================
# # RECOMMENDATIONS
# # =============================================
# print("\n" + "=" * 80)
# print("🎯 ML RECOMMENDATIONS SUMMARY")
# print("=" * 80)

# print("✅ WHAT THIS APPROACH DOES RIGHT:")
# print("• Creates multiple qualifying-focused features")
# print("• Uses cross-validation for robust model selection")
# print("• Provides feature importance analysis")
# print("• Balances qualifying dominance with other factors")
# print("• Uses proper ML validation techniques")

# print("\n🔧 WHY THIS IS BETTER THAN MANUAL WEIGHTING:")
# print("• Model learns optimal feature combinations")
# print("• Captures non-linear relationships")
# print("• More robust to new data")
# print("• Provides interpretable results")

# print("\n📊 KEY INSIGHTS:")
# # print(f"• Qualifying features account for ~{quali_importance/total_importance*100:.0})