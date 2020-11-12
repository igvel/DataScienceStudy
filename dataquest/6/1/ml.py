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

# Predict price based on accomodates
def predict_price(new_listing):
    temp_df = dc_listings.copy()
    temp_df['distance'] = (temp_df['accommodates'] - new_listing).apply(abs)
    mean = temp_df.sort_values('distance').iloc[0:5]['price'].mean()
    return(mean)

acc_one = predict_price(1)
acc_two = predict_price(2)
acc_four = predict_price(4)

# Model validation via train/test sets
train_df = dc_listings.iloc[0:2792]
test_df = dc_listings.iloc[2792:]

def predict_price(new_listing):
    ## DataFrame.copy() performs a deep copy
    temp_df = train_df.copy()
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbor_prices = temp_df.iloc[0:5]['price']
    predicted_price = nearest_neighbor_prices.mean()
    return(predicted_price)

test_df["predicted_price"] = test_df["accommodates"].apply(predict_price)

# Mean absolute error
mae = np.absolute(test_df["price"] - test_df["predicted_price"]).mean()
# Mean squared error
mse = np.square(np.absolute(test_df["price"] - test_df["predicted_price"])).mean()

# Model built on bathrooms
train_df = dc_listings.iloc[0:2792]
test_df = dc_listings.iloc[2792:]

def predict_price(new_listing):
    temp_df = train_df.copy()
    temp_df['distance'] = temp_df['bathrooms'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbors_prices = temp_df.iloc[0:5]['price']
    predicted_price = nearest_neighbors_prices.mean()
    return(predicted_price)

test_df["predicted_price"] = test_df["bathrooms"].apply(predict_price)

squared_error = np.square(numpy.absolute(test_df["price"] - test_df["predicted_price"]))
# Mean squared error
mse = squared_error.mean()
# Root mean square error
rmse = np.sqrt(mse)

# Looking at the ratio of MAE to RMSE can help us understand if there are large but infrequent errors.
errors_one = pd.Series([5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10])
errors_two = pd.Series([5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 1000])

mae_one = errors_one.mean()
mae_two = errors_two.mean()
rmse_one = (errors_one**2).mean()**(1/2)
rmse_two = (errors_two**2).mean()**(1/2)