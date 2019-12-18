# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:21:59 2019

@author: cesreve
"""

# =============================================================================
# librairies
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

# =============================================================================
# data generation
# =============================================================================
data = load_iris()

X = pd.DataFrame(data.data, columns = (data.feature_names))
y = pd.Series(data.target)
# =============================================================================
# bootstrap
# =============================================================================
def bootstrap_replicate_1d(data, func):
    """Generate boostrap replicate of 1D data"""
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)

bs_replicates = np.empty(10000)

for i in range(10000):
    bs_replicates[i] = bootstrap_replicate_1d(X['sepal length (cm)'], np.mean)

# =============================================================================
# plotting a histogram of boostrap replicates
# =============================================================================
_ = plt.hist(bs_replicates, bins = 100, density = True, stacked = True)
_ = plt.xlabel('sepal length mean in cm')
_ = plt.ylabel('PDF')
plt.show()

# =============================================================================
# 
# =============================================================================
