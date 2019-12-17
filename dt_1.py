# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:25:16 2019

@author: wilders    
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris


sns.set()

data = load_iris()

X = pd.DataFrame(data.data, columns = (data.feature_names))
y = pd.Series(data.target)

# =============================================================================
# ECDF
# =============================================================================

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

# Compute ECDF for versicolor data: x_vers, y_vers
x_vers, y_vers = ecdf(X["sepal length (cm)"])

# Generate plot
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')

# Label the axes
_ = plt.xlabel('sepal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
#plt.show()


# =============================================================================
# Percentliles
# =============================================================================

# Specify array of percentiles: percentiles
percentiles = np.asarray([2.5, 25, 50, 75, 97.5])

# Compute percentiles: ptiles_vers
ptiles_sep_len = np.percentile(X["sepal length (cm)"], percentiles)

# Print the result
print(ptiles_sep_len)

# Plot the ECDF
_ = plt.plot(x_vers, y_vers, '.')
_ = plt.xlabel('sepal length (cm)')
_ = plt.ylabel('ECDF')

# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_sep_len,  percentiles/100, marker='D', color='red', linestyle='none')

# Show the plot
plt.show()


# =============================================================================
# Compute variance
# =============================================================================
# Array of differences to mean: differences
differences = X["sepal length (cm)"] - np.mean(X["sepal length (cm)"])

# Square the differences: diff_sq
diff_sq = differences**2

# Compute the mean square difference: variance_explicit
variance_explicit = np.mean(diff_sq)

# Compute the variance using NumPy: variance_np
variance_np = np.var(X["sepal length (cm)"])

# Print the results
print(variance_np, variance_explicit)

# =============================================================================
# Standard deviation
# =============================================================================
# Compute the variance: variance
variance = np.var(X["sepal length (cm)"])

# Print the square root of the variance
print(np.sqrt(variance))

# Print the standard deviation
print(np.std(X["sepal length (cm)"]))


