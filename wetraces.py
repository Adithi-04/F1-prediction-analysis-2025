import fastf1
fastf1.Cache.enable_cache("f1_cache")

def is_wet_race(year, gp_name):
    try:
        session = fastf1.get_session(year, gp_name, "R")
        session.load()
        laps = session.laps
        if "TrackStatus" in laps.columns:
            wet_laps = laps[laps["TrackStatus"] == '6']
            if len(wet_laps) > 10:
                return True
    except Exception as e:
        print(f"Error for {year} {gp_name}: {e}")
    return False

# Example: check all 2022 races
from fastf1.events import get_event_schedule
schedule = get_event_schedule(2022)

wet_races = []
for _, row in schedule.iterrows():
    if is_wet_race(2022, row['EventName']):
        wet_races.append(row['EventName'])

print("\n🌧️ Wet Races in 2022:")
print(wet_races)
