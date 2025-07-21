import pandas as pd
import matplotlib.pyplot as plt

# 2025 Qualifying Data Miami GP
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"],
    "QualifyingTime (s)": [86.983, 86.995, 87.027, 87.299, 87.318, 87.610, 87.822, 87.897, 88.000, 87.836, 88.570, 88.696, 89.271]
})


# Sort qualifying data by qualifying time (ascending)
qualifying_2025_sorted = qualifying_2025.sort_values(by="QualifyingTime (s)").reset_index(drop=True)

# Plotting qualifying times
plt.figure(figsize=(10, 6))
plt.barh(qualifying_2025_sorted["Driver"], qualifying_2025_sorted["QualifyingTime (s)"], color='skyblue')
plt.xlabel('Qualifying Time (s)')
plt.ylabel('Driver')
plt.title('2025 Japanese GP Qualifying Times (Sorted by Track Position)')
plt.gca().invert_yaxis()  
plt.show()
