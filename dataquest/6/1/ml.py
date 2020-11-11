import pandas as pd
import numpy as np

# K-nearest neighbours
dc_listings = pd.read_csv("dc_airbnb.csv")
print(dc_listings.head(1))

# Euclidean distance calculation for 1 feature
first_distance = abs(dc_listings.loc[0, "accommodates"] - 3)
print(first_distance)

distance = (dc_listings["accommodates"] - 3).apply(abs)
dc_listings["distance"] = distance
print(dc_listings["distance"].value_counts())

# Randomize data order, then sort and select first 10 with minimum distance
np.random.seed(1)
randoms = np.random.permutation(dc_listings.shape[0])
dc_listings = dc_listings.loc[randoms]
dc_listings = dc_listings.sort_values("distance")
print(dc_listings.head(10))

# Convert price to float and calculate mean
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')

dc_listings['price'] = stripped_dollars.astype('float')

mean_price = dc_listings.iloc[0:5]['price'].mean()

# Brought along the changes we made to the `dc_listings` Dataframe.
dc_listings = pd.read_csv('dc_airbnb.csv')
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]

# Predict price
def predict_price(new_listing):
    temp_df = dc_listings.copy()
    temp_df['distance'] = (temp_df['accommodates'] - new_listing).apply(abs)
    mean = temp_df.sort_values('distance').iloc[0:5]['price'].mean()
    return(mean)

acc_one = predict_price(1)
acc_two = predict_price(2)
acc_four = predict_price(4)