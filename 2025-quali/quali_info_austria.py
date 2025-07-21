import pandas as pd

# Raw qualifying data
data = [
    {"Position": 1, "Driver": "Lando Norris", "Team": "McLaren", "Time": "1:03.971"},
    {"Position": 2, "Driver": "Charles Leclerc", "Team": "Ferrari", "Time": "1:04.492"},
    {"Position": 3, "Driver": "Oscar Piastri", "Team": "McLaren", "Time": "1:04.554"},
    {"Position": 4, "Driver": "Lewis Hamilton", "Team": "Ferrari", "Time": "1:04.582"},
    {"Position": 5, "Driver": "George Russell", "Team": "Mercedes", "Time": "1:04.763"},
    {"Position": 6, "Driver": "Liam Lawson", "Team": "Racing Bulls", "Time": "1:04.926"},
    {"Position": 7, "Driver": "Max Verstappen", "Team": "Red Bull", "Time": "1:04.929"},
    {"Position": 8, "Driver": "Gabriel Bortoleto", "Team": "Stake", "Time": "1:05.132"},
    {"Position": 9, "Driver": "Kimi Antonelli", "Team": "Mercedes", "Time": "1:05.276"},
    {"Position": 10, "Driver": "Pierre Gasly", "Team": "Alpine", "Time": "1:05.649"},
]

df = pd.DataFrame(data)

def time_to_seconds(t):
    mins, secs = t.split(":")
    return int(mins) * 60 + float(secs)

df["Time (s)"] = df["Time"].apply(time_to_seconds)

print(df)

"""
Result : 
Position             Driver          Team      Time  Time (s)
0         1       Lando Norris       McLaren  1:03.971    63.971
1         2    Charles Leclerc       Ferrari  1:04.492    64.492
2         3      Oscar Piastri       McLaren  1:04.554    64.554
3         4     Lewis Hamilton       Ferrari  1:04.582    64.582
4         5     George Russell      Mercedes  1:04.763    64.763
5         6        Liam Lawson  Racing Bulls  1:04.926    64.926
6         7     Max Verstappen      Red Bull  1:04.929    64.929
7         8  Gabriel Bortoleto         Stake  1:05.132    65.132
8         9     Kimi Antonelli      Mercedes  1:05.276    65.276
9        10       Pierre Gasly        Alpine  1:05.649    65.649
"""