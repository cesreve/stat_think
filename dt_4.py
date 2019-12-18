# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:27:23 2019

@author: cesreve
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


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

# =============================================================================
# Real vs theoritical CDF
# =============================================================================
# Create an ECDF from real data: x, y
x, y = ecdf(X['sepal length (cm)'])

# Create a CDF from theoretical samples: x_theor, y_theor
sepal_len_th = np.random.normal(np.mean(X['sepal length (cm)']),\
                                np.std(X['sepal length (cm)']), size=1500)
x_theor, y_theor = ecdf(sepal_len_th)

# Overlay the plots
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')

# Margins and axis labels
plt.margins(0.02)
plt.xlabel('Sepal lenght')
plt.ylabel('CDF')

# Show the plot
plt.show()

# =============================================================================
# Tests for linear regression
# =============================================================================
plt.plot(X['sepal length (cm)'], X['petal length (cm)'], marker='.', linestyle='none')
plt.show()
