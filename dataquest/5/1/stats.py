import pandas as pd
import matplotlib.pyplot as plt

wnba = pd.read_csv('wnba.csv')

# Simple Random Sampling (SRS)
parameter = wnba["Games Played"].max()
sample = wnba["Games Played"].sample(30, random_state = 1)
statistic = sample.max()
sampling_error = parameter - statistic

# Sampling scatter
mean = wnba["PTS"].mean()
means = []
for i in range(0, 100):
    means.append(wnba["PTS"].sample(10, random_state = i).mean())

plt.scatter(range(1, 101), means)
plt.axhline(mean)

# Stratified sampling
# Max Points per game column
wnba["ppg"] = wnba["PTS"]/wnba["Games Played"]

# Stratification of population
g = wnba[wnba["Pos"] == "G"]
f = wnba[wnba["Pos"] == "F"]
c = wnba[wnba["Pos"] == "C"]
gf = wnba[wnba["Pos"] == "G/F"]
fc = wnba[wnba["Pos"] == "F/C"]

ppgs = {}

for stratum, position in [(g, 'G'), (f, 'F'), (c, 'C'),
                          (gf, 'G/F'), (fc, 'F/C')]:
    sample = stratum.sample(10, random_state = 0)
    ppgs[position] = sample["ppg"].mean()

position_most_points = max(ppgs, key = ppgs.get)

# Stratify by number of games
strata1 = wnba[wnba["Games Played"] <= 12]
strata2 = wnba[(wnba["Games Played"] > 12) & (wnba["Games Played"] <= 22)]
strata3 = wnba[wnba["Games Played"] > 22]

# Sample strata proportionally
means = []
for i in range(0, 100):
    sample1 = strata1.sample(1, random_state = i)
    sample2 = strata2.sample(2, random_state = i)
    sample3 = strata3.sample(7, random_state = i)
    sample = pd.concat([sample1, sample2, sample3])
    means.append(sample["PTS"].mean())

plt.scatter(range(1,101), means)
plt.axhline(wnba["PTS"].mean())

# Cluster sampling
# Select 4 clusters - teams
cluster_names = pd.Series(wnba['Team'].unique()).sample(4, random_state = 0)

clusters = pd.DataFrame()

for cluster in cluster_names:
    clusterdfwnba = wnba.loc[wnba["Team"] == cluster]
    clusters = clusters.append(clusterdfwnba)

height = clusters["Height"].mean()
age = clusters["Age"].mean()
bmi = clusters["BMI"].mean()
totalpts = clusters["PTS"].mean()

sampling_error_height = wnba["Height"].mean() - height
sampling_error_age = wnba["Age"].mean() - age
sampling_error_BMI = wnba["BMI"].mean() - bmi
sampling_error_points = wnba["PTS"].mean() - totalpts