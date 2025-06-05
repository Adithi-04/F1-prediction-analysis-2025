import pandas as pd
import matplotlib.pyplot as plt

# 2025 Qualifying Data Miami GP
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

# Sort qualifying data by qualifying time (ascending)
qualifying_2025_sorted = qualifying_2025.sort_values(by="QualifyingTime (s)").reset_index(drop=True)

# Plotting qualifying times
plt.figure(figsize=(10, 6))
plt.barh(qualifying_2025_sorted["Driver"], qualifying_2025_sorted["QualifyingTime (s)"], color='skyblue')
plt.xlabel('Qualifying Time (s)')
plt.ylabel('Driver')
plt.title('2025 Miami GP Qualifying Times (Sorted by Track Position)')
plt.gca().invert_yaxis()  # To display the fastest driver at the top
plt.show()
