import pandas as pd
import matplotlib.pyplot as plt

# 2025 Qualifying Data Miami GP
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"],
    "QualifyingTime (s)": [90.423, 90.267, 89.841, 90.175, 90.009, 90.772, 90.216, 91.886, 91.303, 90.680, 92.067, 91.886, 92.283]
})


# Sort qualifying data by qualifying time (ascending)
qualifying_2025_sorted = qualifying_2025.sort_values(by="QualifyingTime (s)").reset_index(drop=True)

# Plotting qualifying times
plt.figure(figsize=(10, 6))
plt.barh(qualifying_2025_sorted["Driver"], qualifying_2025_sorted["QualifyingTime (s)"], color='skyblue')
plt.xlabel('Qualifying Time (s)')
plt.ylabel('Driver')
plt.title('2025 Saudi GP Qualifying Times (Sorted by Track Position)')
plt.gca().invert_yaxis()  
plt.show()
