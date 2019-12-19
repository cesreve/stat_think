# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:29:12 2019

@author: cesreve
"""
# =============================================================================
# librairies
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

# =============================================================================
# data generation
# =============================================================================
data = load_iris()

X = pd.DataFrame(data.data, columns = (data.feature_names))
y = pd.Series(data.target)

target_mapper =  {0: "Setosa", 1: "Versicolour", 2:"Virginica"}
y.replace(target_mapper, inplace = True)
print(y)

iris = pd.DataFrame(pd.concat([X,y], axis = 1))
iris.columns = ['sepal length (cm)',  'sepal width (cm)', 'petal length (cm)',
                               'petal width (cm)', 'species']
iris.head()

dataset = iris[iris["species"].isin(['Versicolour','Virginica'])][['sepal length (cm)', 'species']]
dataset.head()

sep_len_vers = np.ravel(iris[iris['species']=='Versicolour'][["sepal length (cm)"]])
sep_len_virg = data.data[:,0][100:]
#g = sns.pairplot(iris, hue="species")

# =============================================================================
# ECDF definition
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
# Plotting ECDFs
# =============================================================================
x_vers, y_vers = ecdf(np.ravel(iris[iris['species']=='Versicolour'][["sepal length (cm)"]]))
x_virg, y_virg = ecdf(np.ravel(iris[iris['species']=='Virginica'][["sepal length (cm)"]]))

# Generate plot
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none', color = 'green')
_ = plt.plot(x_virg, y_virg, marker='.', linestyle='none', color = 'red')

# Label the axes
_ = plt.xlabel('sepal length (cm)')
_ = plt.ylabel('ECDF')

plt.show()

# =============================================================================
# Testing difference between Versicolour and Virginica on 'sepal length (cm)'
# =============================================================================
def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate(data1, data2)

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permuation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[len(data1):]
    perm_sample_2 = permuted_data[:len(data1)]

    return perm_sample_1, perm_sample_2

permutation_sample(np.ravel(iris[iris['species']=='Versicolour'][["sepal length (cm)"]]),\
                   np.ravel(iris[iris['species']=='Virginica'][["sepal length (cm)"]]))
np.ravel(iris[iris['species']=='Versicolour'][["sepal length (cm)"]])

np.concatenate(np.ravel(iris[iris['species']=='Versicolour'][["sepal length (cm)"]]),\
                   np.ravel(iris[iris['species']=='Virginica'][["sepal length (cm)"]]))







