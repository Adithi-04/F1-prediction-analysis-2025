
# 🏎️ Formula 1 Race Outcome Predictor

This project leverages machine learning and historical race data to predict race winners, podium finishers, and race performance across Formula 1 Grand Prix weekends.

## 🚀 Overview

Built using `FastF1`, `scikit-learn`, and Python, the model analyzes driver performance based on:

* Sector times from previous races
* Qualifying results from the current season
* Driver-specific metrics (e.g. wet-weather skill, season points)

Predictions are generated for all races and visualized with feature importance to understand what influences race performance the most.

## 📊 Features

* Predicts race winners and podium finishers for upcoming GPs
* Uses Gradient Boosting, Ridge/Lasso, and XGBoost models
* Real-time telemetry via FastF1
* Advanced feature engineering with driver form and qualifying pace
* Clean, interpretable visualizations of predictions and model weights
* * Added real-time weather impact

## 📦 Tech Stack

* Python, Pandas, NumPy, Matplotlib
* FastF1 for telemetry and lap data
* Scikit-learn (GradientBoosting, Ridge, Lasso)
* XGBoost for alternative modeling

## 🛠 How to Run

1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/f1-race-predictor.git
   cd f1-race-predictor
   ```

2. Create a virtual environment and install requirements

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run prediction script

   ```bash
   python 2025-gp/4.bahraingp.py
   ```

## 📌 Dataset Sources

* FastF1 (`https://theoehrly.github.io/Fast-F1/`)
* 2025 qualifying data



