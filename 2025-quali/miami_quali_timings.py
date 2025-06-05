from fastf1 import get_session
import fastf1
import pandas as pd

# Enable cache
fastf1.Cache.enable_cache("f1_cache")

# Load Qualifying session
session = get_session(2025, "Miami", "Q")
session.load()

# Get all laps
laps = session.laps

# Get each driver's fastest qualifying lap
fastest_laps = laps.pick_quicklaps().groupby("Driver").apply(lambda x: x.pick_fastest()).reset_index(drop=True)

# Build DataFrame with driver and best time
qualifying_times = fastest_laps[["Driver", "LapTime"]].copy()
qualifying_times["QualifyingTime (s)"] = qualifying_times["LapTime"].dt.total_seconds()

# Print it
print(qualifying_times.sort_values("QualifyingTime (s)"))
