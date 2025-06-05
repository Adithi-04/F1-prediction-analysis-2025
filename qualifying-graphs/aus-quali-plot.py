import pandas as pd
import matplotlib.pyplot as plt

# 2025 Qualifying Data Miami GP
qualifying_2025 = pd.DataFrame({
    "Driver": ["Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell", 
               "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
               "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"],
    "QualifyingTime (s)": [75.096, 75.180, 75.481, 75.546, 75.670, 75.737,
                           75.755, 75.973, 75.980, 76.062, 76.4, 76.5]
})


# Sort qualifying data by qualifying time (ascending)
qualifying_2025_sorted = qualifying_2025.sort_values(by="QualifyingTime (s)").reset_index(drop=True)

# Plotting qualifying times
plt.figure(figsize=(10, 6))
plt.barh(qualifying_2025_sorted["Driver"], qualifying_2025_sorted["QualifyingTime (s)"], color='skyblue')
plt.xlabel('Qualifying Time (s)')
plt.ylabel('Driver')
plt.title('2025 Australian GP Qualifying Times (Sorted by Track Position)')
plt.gca().invert_yaxis()  # To display the fastest driver at the top
plt.show()
