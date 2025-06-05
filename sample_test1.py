import fastf1
import pandas as pd
fastf1.Cache.enable_cache('./f1_cache')

# def get_race_winner(year, race_name):
#     session = fastf1.get_session(year, race_name, 'R')
#     session.load()

#     results = session.results
#     print("Available columns:", results.columns)
#     print(results.head())  # Peek at the first few rows

#     return {
#         'year': year,
#         'race': race_name,
#         'winner': 'TBD'
#     }


# race_result = get_race_winner(2024, 'Monaco')

session_2022= fastf1.get_session(2025, "Bahrain", "Q")
session_2022.load()
results=session_2022.results
race_times_2022 = session_2022.results[["Abbreviation", "Time"]]
print(race_times_2022)

