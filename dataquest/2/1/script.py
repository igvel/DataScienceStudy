import csv
import numpy as np

# 1D array
data_ndarray = np.array([10, 20, 30])

# import nyc_taxi.csv as a list of lists
f = open("nyc_taxis.csv", "r")
taxi_list = list(csv.reader(f))

# remove the header row
taxi_list = taxi_list[1:]

# convert all values to floats
converted_taxi_list = []
for row in taxi_list:
    converted_row = []
    for item in row:
        converted_row.append(float(item))
    converted_taxi_list.append(converted_row)

# 2D array
taxi = np.array(converted_taxi_list)
# Shape == tuple (nrows, ncols)
taxi_shape = taxi.shape

print(taxi_shape)

# Selecting subarrays
row_0 = taxi[0]
rows_391_to_500 = taxi[391:501]
row_21_column_5 = taxi[21, 5]

columns_1_4_7 = taxi[:, [1, 4, 7]]
row_99_columns_5_to_8 = taxi[99, 5:9]
rows_100_to_200_column_14 = taxi[100:201, 14]

# Vector operations
fare_amount = taxi[:, 9]
fees_amount = taxi[:, 10]

fare_and_fees = fare_amount + fees_amount

trip_distance_miles = taxi[:, 7]
trip_length_seconds = taxi[:, 8]

trip_length_hours = trip_length_seconds / 3600  # 3600 seconds is one hour

trip_mph = trip_distance_miles / trip_length_hours

# Statistics
mph_min = trip_mph.min()
mph_max = trip_mph.max()
mph_mean = trip_mph.mean()

# we'll compare against the first 5 rows only
taxi_first_five = taxi[:5]
# select these columns: fare_amount, fees_amount, tolls_amount, tip_amount
fare_components = taxi_first_five[:, 9:13]

# Sum by row
fare_sums = fare_components.sum(axis=1)
fare_totals = taxi_first_five[:, 13]
print(fare_totals, fare_sums)

# Read ND array from file
taxi = np.genfromtxt('nyc_taxis.csv', delimiter=',', skip_header=1)
taxi_shape = taxi.shape
# Data type
print(taxi.dtype)

# Boolean arrays
a = np.array([1, 2, 3, 4, 5])
b = np.array(["blue", "blue", "red", "blue"])
c = np.array([80.0, 103.4, 96.9, 200.3])

a_bool = a < 3
b_bool = b == "blue"
c_bool = c > 100

# Boolean indexing - act like a filter
pickup_month = taxi[:, 1]

# Select rides in february
february_bool = pickup_month == 2
february = pickup_month[february_bool]
february_rides = february.shape[0]

# Top tips
tip_amount = taxi[:, 12]
tip_bool = tip_amount > 50
top_tips = taxi[tip_bool, 5:14]

# this creates a copy of our taxi ndarray
taxi_modified = taxi.copy()
# Modifications
taxi_modified[28214, 5] = 1
taxi_modified[:, 0] = 16
taxi_modified[1800:1802, 7] = taxi_modified[:, 7].mean()

# this creates a copy of our taxi ndarray
taxi_copy = taxi.copy()
# Modification using boolean indexing
total_amount = taxi_copy[:, 13]
taxi_copy[total_amount < 0] = 0

# create a new column filled with `0`.
zeros = np.zeros([taxi.shape[0], 1])
# Concatenate arrays by axis
taxi_modified = np.concatenate([taxi, zeros], axis=1)
print(taxi_modified)

taxi_modified[taxi_modified[:, 5] == 2, 15] = 1
taxi_modified[taxi_modified[:, 5] == 3, 15] = 1
taxi_modified[taxi_modified[:, 5] == 5, 15] = 1

# Calculate counts
jfk = taxi[taxi[:, 6] == 2]
jfk_count = jfk.shape[0]
laguardia = taxi[taxi[:, 6] == 3]
laguardia_count = laguardia.shape[0]
newark = taxi[taxi[:, 6] == 5]
newark_count = newark.shape[0]

# Clean up data and calculate means
trip_mph = taxi[:,7] / (taxi[:,8] / 3600)

cleaned_taxi = taxi[trip_mph < 100]
mean_distance = cleaned_taxi[:, 7].mean()
mean_length = cleaned_taxi[:, 8].mean()
mean_total_amount = cleaned_taxi[:, 13].mean()
