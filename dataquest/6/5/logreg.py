# Logistic regression

import pandas as pd
import matplotlib.pyplot as plt
admissions = pd.read_csv("admissions.csv")
plt.scatter(admissions['gpa'], admissions['admit'])
plt.show()


import numpy as np

# Logistic Function
def logistic(x):
    # np.exp(x) raises x to the exponential power, ie e^x. e ~= 2.71828
    return np.exp(x)  / (1 + np.exp(x))

# Generate 50 real values, evenly spaced, between -6 and 6.
x = np.linspace(-6,6,50, dtype=float)

# Transform each number in t using the logistic function.
y = logistic(x)

# Plot the resulting data.
plt.plot(x, y)
plt.ylabel("Probability")
plt.show()

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
# Fit model
logistic_model.fit(admissions[["gpa"]], admissions["admit"])

# Predict probability
pred_probs = logistic_model.predict_proba(admissions[["gpa"]])

# Show probability correlation w/ gpa  - it's linear!
plt.scatter(admissions["gpa"], pred_probs[:,1])

# Predict label - 0 or 1
fitted_labels = logistic_model.predict(admissions[["gpa"]])

plt.scatter(admissions["gpa"], fitted_labels)#