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
    return (mean)

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
    return (predicted_price)


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
    return (predicted_price)


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
rmse_one = (errors_one ** 2).mean() ** (1 / 2)
rmse_two = (errors_two ** 2).mean() ** (1 / 2)

# drop columns
drop_columns = ['room_type', 'city', 'state', 'latitude', 'longitude', 'zipcode', 'host_response_rate',
                'host_acceptance_rate', 'host_listings_count']
dc_listings = dc_listings.drop(drop_columns, axis=1)
# List number of null values
print(dc_listings.isnull().sum())

# Drop columns with many nulls
drop_columns = ['cleaning_fee', 'security_deposit']
dc_listings = dc_listings.drop(drop_columns, axis=1)
# Drop rows with null values
dc_listings = dc_listings.dropna()
print(dc_listings.isnull().sum())

# Normalize data - bring to normal distribution with mean = 0, and std deviation 1
normalized_listings = (dc_listings - dc_listings.mean()) / (dc_listings.std())
normalized_listings["price"] = dc_listings["price"]
print(normalized_listings.head(3))

from scipy.spatial import distance

# Calculate euclidean distance between 0 and 4 rows
first_fifth_distance = distance.euclidean(normalized_listings.iloc[0][["accommodates", "bathrooms"]],
                                          normalized_listings.iloc[4][["accommodates", "bathrooms"]])

# Use SciKit-learn for kNN regression model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

train_df = normalized_listings.iloc[0:2792]
test_df = normalized_listings.iloc[2792:]

train_columns = ['accommodates', 'bathrooms']
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute', metric='euclidean')
# Train (In case of KNN - nothing to train, just store data for prediction calculation)
knn.fit(train_df[train_columns], train_df['price'])
# Predict
predictions = knn.predict(test_df[train_columns])
# Evaluate performance
two_features_mse = mean_squared_error(test_df["price"], predictions)
two_features_rmse = two_features_mse**(1/2)

# Train on all features excepting target
features = train_df.columns.tolist()
features.remove('price')
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')

# Train (In case of KNN - nothing to train, just store data for prediction calculation)
knn.fit(train_df[features], train_df['price'])
# Predict
all_features_predictions = knn.predict(test_df[features])
# Evaluate performance
all_features_mse = mean_squared_error(test_df["price"], all_features_predictions)
all_features_rmse = all_features_mse**(1/2)

# Hyper params (k) grid search
hyper_params = range(1,21)
mse_values = []
for h in hyper_params:
    knn = KNeighborsRegressor(n_neighbors = h, algorithm = 'brute')
    train_columns = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
    knn.fit(train_df[train_columns], train_df['price'])
    # Predict
    predictions = knn.predict(test_df[train_columns])
    # Evaluate performance
    mse_values.append(mean_squared_error(test_df["price"], predictions))

print(mse_values)

import matplotlib.pyplot as plt
# Show trend
plt.scatter(hyper_params, mse_values)
plt.show()

# Overall workflow to select best k
two_features = ['accommodates', 'bathrooms']
three_features = ['accommodates', 'bathrooms', 'bedrooms']
hyper_params = [x for x in range(1,21)]
# Append the first model's MSE values to this list.
two_mse_values = list()
# Append the second model's MSE values to this list.
three_mse_values = list()
two_hyp_mse = dict()
three_hyp_mse = dict()

for h in hyper_params:
    knn = KNeighborsRegressor(n_neighbors = h, algorithm = 'brute')
    knn.fit(train_df[two_features], train_df['price'])
    # Predict
    predictions = knn.predict(test_df[two_features])
    # Evaluate performance
    two_mse_values.append(mean_squared_error(test_df["price"], predictions))

for h in hyper_params:
    knn = KNeighborsRegressor(n_neighbors = h, algorithm = 'brute')
    knn.fit(train_df[three_features], train_df['price'])
    # Predict
    predictions = knn.predict(test_df[three_features])
    # Evaluate performance
    three_mse_values.append(mean_squared_error(test_df["price"], predictions))

two_lowest_mse = min(two_mse_values)
for (k, mse) in enumerate(two_mse_values, start = 1):
    if mse == two_lowest_mse:
        two_hyp_mse[k] = mse
        break

three_lowest_mse = min(three_mse_values)
for (k, mse) in enumerate(three_mse_values, start = 1):
    if mse == three_lowest_mse:
        three_hyp_mse[k] = mse
        break


# Holdout validation
import numpy as np
import pandas as pd

dc_listings = pd.read_csv("dc_airbnb.csv")
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
# Shuffle data
dc_listings = dc_listings.reindex(np.random.permutation(dc_listings.index))

# Split into 2 sets
split_one = dc_listings.iloc[0:1862].copy()
split_two = dc_listings.iloc[1862:].copy()

train_one = split_one
test_one = split_two
train_two = split_two
test_two = split_one
# First half
model = KNeighborsRegressor()
model.fit(train_one[["accommodates"]], train_one["price"])
test_one["predicted_price"] = model.predict(test_one[["accommodates"]])
iteration_one_rmse = mean_squared_error(test_one["price"], test_one["predicted_price"])**(1/2)

# Second half
model.fit(train_two[["accommodates"]], train_two["price"])
test_two["predicted_price"] = model.predict(test_two[["accommodates"]])
iteration_two_rmse = mean_squared_error(test_two["price"], test_two["predicted_price"])**(1/2)

avg_rmse = np.mean([iteration_two_rmse, iteration_one_rmse])

print(iteration_one_rmse, iteration_two_rmse, avg_rmse)

# k-fold validation
dc_listings.loc[dc_listings.index[0:745], "fold"] = 1
dc_listings.loc[dc_listings.index[745:1490], "fold"] = 2
dc_listings.loc[dc_listings.index[1490:2234], "fold"] = 3
dc_listings.loc[dc_listings.index[2234:2978], "fold"] = 4
dc_listings.loc[dc_listings.index[2978:3723], "fold"] = 5

print(dc_listings['fold'].value_counts())
print("\n Num of missing values: ", dc_listings['fold'].isnull().sum())

fold_ids = [1,2,3,4,5]
def train_and_validate(df, folds):
    fold_rmses = []
    for fold in folds:
        # Train
        model = KNeighborsRegressor()
        train = df[df["fold"] != fold]
        test = df[df["fold"] == fold].copy()
        model.fit(train[["accommodates"]], train["price"])
        # Predict
        labels = model.predict(test[["accommodates"]])
        test["predicted_price"] = labels
        mse = mean_squared_error(test["price"], test["predicted_price"])
        rmse = mse**(1/2)
        fold_rmses.append(rmse)
    return(fold_rmses)

rmses = train_and_validate(dc_listings, fold_ids)
print(rmses)
avg_rmse = np.mean(rmses)
print(avg_rmse)

# k-fold validation using SciKit
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(5, shuffle=True, random_state=1)
model = KNeighborsRegressor()
mses = cross_val_score(model, dc_listings[["accommodates"]], dc_listings["price"], scoring="neg_mean_squared_error", cv=kf)
rmses = np.sqrt(np.absolute(mses))
avg_rmse = np.mean(rmses)
print(rmses)
print(avg_rmse)

num_folds = [3, 5, 7, 9, 10, 11, 13, 15, 17, 19, 21, 23]

for fold in num_folds:
    kf = KFold(fold, shuffle=True, random_state=1)
    model = KNeighborsRegressor()
    mses = cross_val_score(model, dc_listings[["accommodates"]], dc_listings["price"], scoring="neg_mean_squared_error", cv=kf)
    rmses = np.sqrt(np.absolute(mses))
    avg_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)
    print(str(fold), "folds: ", "avg RMSE: ", str(avg_rmse), "std RMSE: ", str(std_rmse))