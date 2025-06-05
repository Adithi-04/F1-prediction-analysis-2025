def time_to_seconds(time_str):
    # Converts time "m:ss.sss" to total seconds as float
    minutes, seconds = time_str.split(':')
    return int(minutes)*60 + float(seconds)

data = [
    {"Driver": "Oscar Piastri", "Q1": "1:31.392", "Q2": "1:30.454", "Q3": "1:29.841"},
    {"Driver": "George Russell", "Q1": "1:31.494", "Q2": "1:30.664", "Q3": "1:30.009"},
    {"Driver": "Charles Leclerc", "Q1": "1:31.454", "Q2": "1:30.724", "Q3": "1:30.175"},
    {"Driver": "Kimi Antonelli", "Q1": "1:31.415", "Q2": "1:30.716", "Q3": "1:30.213"},
    {"Driver": "Pierre Gasly", "Q1": "1:31.462", "Q2": "1:30.643", "Q3": "1:30.216"},
    {"Driver": "Lando Norris", "Q1": "1:31.107", "Q2": "1:30.560", "Q3": "1:30.267"},
    {"Driver": "Max Verstappen", "Q1": "1:31.303", "Q2": "1:31.019", "Q3": "1:30.423"},
    {"Driver": "Carlos Sainz", "Q1": "1:31.591", "Q2": "1:30.844", "Q3": "1:30.680"},
    {"Driver": "Lewis Hamilton", "Q1": "1:31.219", "Q2": "1:31.009", "Q3": "1:30.772"},
    {"Driver": "Yuki Tsunoda", "Q1": "1:31.751", "Q2": "1:31.228", "Q3": "1:31.303"},
    {"Driver": "Jack Doohan", "Q1": "1:31.414", "Q2": "1:31.245", "Q3": None},
    {"Driver": "Isack Hadjar", "Q1": "1:31.591", "Q2": "1:31.271", "Q3": None},
    {"Driver": "Fernando Alonso", "Q1": "1:31.634", "Q2": "1:31.886", "Q3": None},
    {"Driver": "Esteban Ocon", "Q1": "1:31.594", "Q2": None, "Q3": None},
    {"Driver": "Alexander Albon", "Q1": "1:32.040", "Q2": None, "Q3": None},
    {"Driver": "Nico Hulkenberg", "Q1": "1:32.067", "Q2": None, "Q3": None},
    {"Driver": "Liam Lawson", "Q1": "1:32.165", "Q2": None, "Q3": None},
    {"Driver": "Gabriel Bortoleto", "Q1": "1:32.186", "Q2": None, "Q3": None},
    {"Driver": "Lance Stroll", "Q1": "1:32.283", "Q2": None, "Q3": None},
    {"Driver": "Oliver Bearman", "Q1": "1:32.373", "Q2": None, "Q3": None},
]

for driver in data:
    times = [time_to_seconds(t) for t in [driver["Q1"], driver["Q2"], driver["Q3"]] if t is not None]
    min_time = min(times)
    print(f"{driver['Driver']}: {min_time:.3f} seconds")
