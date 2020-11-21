## Decision trees
import pandas
# Load data on income of US citizen and their demographics
# Set index_col to False to avoid pandas thinking that the first column is row indexes (it's age)
income = pandas.read_csv("income.csv", index_col=False)
print(income.head(5))

# Convert columns from text categories to numbers
col = pandas.Categorical(income["workclass"])
income["workclass"] = col.codes
print(income["workclass"].head(5))
for name in ["education", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "high_income"]:
    col = pandas.Categorical(income[name])
    income[name] = col.codes

# Split data on workers into Private (4) and Non-Private sectors
private_incomes = income[income["workclass"] == 4]
public_incomes = income[income["workclass"] != 4]

print(private_incomes.shape)
print(public_incomes.shape)

import math
# Calculating entropy
prob_0 = income[income["high_income"] == 0].shape[0] / income.shape[0]
prob_1 = income[income["high_income"] == 1].shape[0] / income.shape[0]
income_entropy = -(prob_0 * math.log(prob_0, 2) + prob_1 * math.log(prob_1, 2))

import numpy

def calc_entropy(column):
    """
    Calculate entropy given a pandas series, list, or numpy array.
    """
    # Compute the counts of each unique value in the column
    counts = numpy.bincount(column)
    # Divide by the total column length to get a probability
    probabilities = counts / len(column)

    # Initialize the entropy to 0
    entropy = 0
    # Loop through the probabilities, and add each one to the total entropy
    for prob in probabilities:
        if prob > 0:
            entropy += prob * math.log(prob, 2)

    return -entropy

# Verify that our function matches our answer from earlier
entropy = calc_entropy([1,1,0,0,1])
print(entropy)

# Calculate Information Gain
information_gain = entropy - ((.8 * calc_entropy([1,1,0,0])) + (.2 * calc_entropy([1])))
print(information_gain)
income_entropy = calc_entropy(income["high_income"])

median_age = income["age"].median()

left_split = income[income["age"] <= median_age]
right_split = income[income["age"] > median_age]

age_information_gain = income_entropy - ((left_split.shape[0] / income.shape[0]) * calc_entropy(left_split["high_income"])
                                         + ((right_split.shape[0] / income.shape[0]) * calc_entropy(right_split["high_income"])))

def calc_information_gain(data, split_name, target_name):
    """
    Calculate information gain given a data set, column to split on, and target
    """
    # Calculate the original entropy
    original_entropy = calc_entropy(data[target_name])

    # Find the median of the column we're splitting
    column = data[split_name]
    median = column.median()

    # Make two subsets of the data, based on the median
    left_split = data[column <= median]
    right_split = data[column > median]

    # Loop through the splits and calculate the subset entropies
    to_subtract = 0
    for subset in [left_split, right_split]:
        prob = (subset.shape[0] / data.shape[0])
        to_subtract += prob * calc_entropy(subset[target_name])

    # Return information gain
    return original_entropy - to_subtract

columns = ["age", "workclass", "education_num", "marital_status",
           "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

information_gains = []
for col in columns:
    information_gains.append(calc_information_gain(income, col, "high_income"))

# Calculate column with highest Information Gain
highest_gain_index = information_gains.index(max(information_gains))
highest_gain = columns[highest_gain_index]

# ID3 Algorithm for Tree
# A list of columns to potentially split income with
columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]
def find_best_column(data, target_name, columns):
    information_gains = []
    for col in columns:
        information_gain = calc_information_gain(data, col, "high_income")
        information_gains.append(information_gain)
    # Find the name of the column with the highest gain
    highest_gain_index = information_gains.index(max(information_gains))
    highest_gain = columns[highest_gain_index]
    return highest_gain

income_split = find_best_column(income, "high_income", columns)

# We'll use lists to store our labels for nodes (when we find them)
# Lists can be accessed inside our recursive function, whereas integers can't.
# Look at the python missions on scoping for more information on this topic
label_1s = []
label_0s = []

def id3(data, target, columns):
    # The pandas.unique method will return a list of all the unique values in a series
    unique_targets = pandas.unique(data[target])

    if len(unique_targets) == 1:
        # Insert code here to append 1 to label_1s or 0 to label_0s, based on what we should label the node
        if 0 in unique_targets:
            label_0s.append(0)
        elif 1 in unique_targets:
            label_1s.append(1)
        # Returning here is critical -- if we don't, the recursive tree will never finish, and run forever
        # See our example above for when we returned
        return

    # Find the best column to split on in our data
    best_column = find_best_column(data, target, columns)
    # Find the median of the column
    column_median = data[best_column].median()

    # Create the two splits
    left_split = data[data[best_column] <= column_median]
    right_split = data[data[best_column] > column_median]

    # Loop through the splits and call id3 recursively
    for split in [left_split, right_split]:
        # Call id3 recursively to process each branch
        id3(split, target, columns)

# Create the data set that we used in the example on the last screen
data = pandas.DataFrame([
    [0,20,0],
    [0,60,2],
    [0,40,1],
    [1,25,1],
    [1,35,2],
    [1,55,1]
])
# Assign column names to the data
data.columns = ["high_income", "age", "marital_status"]

# Call the function on our data to set the counters properly
id3(data, "high_income", ["age", "marital_status"])

# Tree construction
tree = {}
nodes = []

def id3(data, target, columns, tree):
    unique_targets = pandas.unique(data[target])
    nodes.append(len(nodes) + 1)
    tree["number"] = nodes[-1]

    if len(unique_targets) == 1:
        if 0 in unique_targets:
            tree["label"] = 0
        elif 1 in unique_targets:
            tree["label"] = 1
        return

    best_column = find_best_column(data, target, columns)
    column_median = data[best_column].median()

    tree["column"] = best_column
    tree["median"] = column_median

    left_split = data[data[best_column] <= column_median]
    right_split = data[data[best_column] > column_median]
    split_dict = [["left", left_split], ["right", right_split]]

    for name, split in split_dict:
        tree[name] = {}
        id3(split, target, columns, tree[name])

id3(data, "high_income", ["age", "marital_status"], tree)

# Print tree outline
def print_with_depth(string, depth):
    # Add space before a string
    prefix = "    " * depth
    # Print a string, and indent it appropriately
    print("{0}{1}".format(prefix, string))

def print_node(tree, depth):
    if "label" in tree:
        print_with_depth("Leaf: Label {0}".format(tree["label"]), depth)
        return
    print_with_depth("{0} > {1}".format(tree["column"], tree["median"]), depth)
    for branch in [tree["left"], tree["right"]]:
        print_node(branch, depth+1)

print_node(tree, 0)

# Prediction using tree
def predict(tree, row):
    if "label" in tree:
        return tree["label"]

    column = tree["column"]
    median = tree["median"]
    if row[column] <= median:
        return predict(tree["left"], row)
    else:
        return predict(tree["right"], row)

print(predict(tree, data.iloc[0]))

# Batch prediction
new_data = pandas.DataFrame([
    [40,0],
    [20,2],
    [80,1],
    [15,1],
    [27,2],
    [38,1]
])
# Assign column names to the data
new_data.columns = ["age", "marital_status"]

def batch_predict(tree, df):
    return df.apply(lambda row: predict(tree, row), axis=1)

predictions = batch_predict(tree, new_data)

# DT-Classifier
from sklearn.tree import DecisionTreeClassifier

# A list of columns to train with
# We've already converted all columns to numeric
columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

# Instantiate the classifier
# Set random_state to 1 to make sure the results are consistent
clf = DecisionTreeClassifier(random_state=1)

# We've already loaded the variable "income," which contains all of the income data
clf.fit(income[columns], income["high_income"])

# Train/test split
import numpy
import math

# Set a random seed so the shuffle is the same every time
numpy.random.seed(1)

# Shuffle the rows
# This permutes the index randomly using numpy.random.permutation
# Then, it reindexes the dataframe with the result
# The net effect is to put the rows into random order
income = income.reindex(numpy.random.permutation(income.index))

train_max_row = math.floor(income.shape[0] * .8)
train = income.iloc[:train_max_row]
test = income.iloc[train_max_row:]

# Calculate ROC AUC score to estimate error of prediction
from sklearn.metrics import roc_auc_score

clf = DecisionTreeClassifier(random_state=1)
clf.fit(train[columns], train["high_income"])

predictions = clf.predict(test[columns])
# AUC on test data
error = roc_auc_score(test["high_income"], predictions)
print(error)
# AUC on train data
predictions = clf.predict(train[columns])
print(roc_auc_score(train["high_income"], predictions))

# Limit number of samples in the node to split to prevent overfitting
clf = DecisionTreeClassifier(min_samples_split=13, random_state=1)
clf.fit(train[columns], train["high_income"])
predictions = clf.predict(test[columns])
test_auc = roc_auc_score(test["high_income"], predictions)

train_predictions = clf.predict(train[columns])
train_auc = roc_auc_score(train["high_income"], train_predictions)

print(test_auc)
print(train_auc)

# Limit depth of the tree to prevent overfitting
clf = DecisionTreeClassifier(random_state=1, min_samples_split=13, max_depth=7)
clf.fit(train[columns], train["high_income"])
predictions = clf.predict(test[columns])
test_auc = roc_auc_score(test["high_income"], predictions)

train_predictions = clf.predict(train[columns])
train_auc = roc_auc_score(train["high_income"], train_predictions)

print(test_auc)
print(train_auc)

# Noise can cause overfitting of the model due to variance
numpy.random.seed(1)

# Generate a column containing random numbers from 0 to 4
income["noise"] = numpy.random.randint(4, size=income.shape[0])

# Adjust "columns" to include the noise column
columns = ["noise", "age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

# Make new train and test sets
train_max_row = math.floor(income.shape[0] * .8)
train = income.iloc[:train_max_row]
test = income.iloc[train_max_row:]

# Initialize the classifier
clf = DecisionTreeClassifier(random_state=1)
clf.fit(train[columns], train["high_income"])
predictions = clf.predict(test[columns])
test_auc = roc_auc_score(test["high_income"], predictions)

train_predictions = clf.predict(train[columns])
train_auc = roc_auc_score(train["high_income"], train_predictions)

print(test_auc)
print(train_auc)

## Random Forests
# Compare two trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=2)
clf.fit(train[columns], train["high_income"])

clf2 = DecisionTreeClassifier(random_state=1, max_depth=5)
clf2.fit(train[columns], train["high_income"])
predictions = clf.predict(test[columns])
print(roc_auc_score(test["high_income"], predictions))

predictions = clf2.predict(test[columns])
print(roc_auc_score(test["high_income"], predictions))

# Combine trees using mean of probabilities
predictions = clf.predict_proba(test[columns])[:,1]
predictions2 = clf2.predict_proba(test[columns])[:,1]
combined = (predictions + predictions2) / 2
rounded = numpy.round(combined)

print(roc_auc_score(test["high_income"], rounded))

# Use bagging method of Random Forest
# We'll build 10 trees
tree_count = 10

# Each "bag" will have 60% of the number of original rows
bag_proportion = .6

predictions = []
for i in range(tree_count):
    # We select 60% of the rows from train, sampling with replacement
    # We set a random state to ensure we'll be able to replicate our results
    # We set it to i instead of a fixed value so we don't get the same sample in every loop
    # That would make all of our trees the same
    bag = train.sample(frac=bag_proportion, replace=True, random_state=i)

    # Fit a decision tree model to the "bag"
    clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=2)
    clf.fit(bag[columns], bag["high_income"])

    # Using the model, make predictions on the test data
    predictions.append(clf.predict_proba(test[columns])[:,1])

combined = numpy.sum(predictions, axis=0) / 10
rounded = numpy.round(combined)

print(roc_auc_score(test["high_income"], rounded))

# Random feature selection for ID3
# Create the data set that we used two missions ago
data = pandas.DataFrame([
    [0,4,20,0],
    [0,4,60,2],
    [0,5,40,1],
    [1,4,25,1],
    [1,5,35,2],
    [1,5,55,1]
])
data.columns = ["high_income", "employment", "age", "marital_status"]

# Set a random seed to make the results reproducible
numpy.random.seed(1)

# The dictionary to store our tree
tree = {}
nodes = []

# The function to find the column to split on
def find_best_column(data, target_name, columns):
    information_gains = []

    # Select two columns randomly
    cols = numpy.random.choice(columns, 2)

    for col in cols:
        information_gain = calc_information_gain(data, col, "high_income")
        information_gains.append(information_gain)

    highest_gain_index = information_gains.index(max(information_gains))

    # Get the highest gain by indexing "cols"
    highest_gain = cols[highest_gain_index]
    return highest_gain

# The function to construct an ID3 decision tree
def id3(data, target, columns, tree):
    unique_targets = pandas.unique(data[target])
    nodes.append(len(nodes) + 1)
    tree["number"] = nodes[-1]

    if len(unique_targets) == 1:
        if 0 in unique_targets:
            tree["label"] = 0
        elif 1 in unique_targets:
            tree["label"] = 1
        return

    best_column = find_best_column(data, target, columns)
    column_median = data[best_column].median()

    tree["column"] = best_column
    tree["median"] = column_median

    left_split = data[data[best_column] <= column_median]
    right_split = data[data[best_column] > column_median]
    split_dict = [["left", left_split], ["right", right_split]]

    for name, split in split_dict:
        tree[name] = {}
        id3(split, target, columns, tree[name])


# Run the ID3 algorithm on our data set and print the resulting tree
id3(data, "high_income", ["employment", "age", "marital_status"], tree)
print(tree)

# Use Random feature subset together with bagging for Random forest
# We'll build 10 trees
tree_count = 10

# Each "bag" will have 60% of the number of original rows
bag_proportion = .6

predictions = []
for i in range(tree_count):
    # We select 60% of the rows from train, sampling with replacement
    # We set a random state to ensure we'll be able to replicate our results
    # We set it to i instead of a fixed value so we don't get the same sample every time
    bag = train.sample(frac=bag_proportion, replace=True, random_state=i)

    # Fit a decision tree model to the "bag"
    clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=2, splitter="random", max_features="auto")
    clf.fit(bag[columns], bag["high_income"])

    # Using the model, make predictions on the test data
    predictions.append(clf.predict_proba(test[columns])[:,1])

combined = numpy.sum(predictions, axis=0) / 10
rounded = numpy.round(combined)

print(roc_auc_score(test["high_income"], rounded))

# Random Forest using SciKit-learn
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=5, random_state=1, min_samples_leaf=2)
clf.fit(train[columns], train["high_income"])

predictions = clf.predict(test[columns])
print(roc_auc_score(test["high_income"], predictions))

# Check fo overfit
clf = RandomForestClassifier(n_estimators=150, random_state=1, min_samples_leaf=5)

clf.fit(train[columns], train["high_income"])

predictions = clf.predict(train[columns])
print(roc_auc_score(train["high_income"], predictions))

predictions = clf.predict(test[columns])
print(roc_auc_score(test["high_income"], predictions))