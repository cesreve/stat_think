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

sep_len_vers = data.data[:,0][51:99]
#sep_len_vers = sep_len_vers.reshape((1, -1))

sep_len_virg = data.data[:,0][100:]
#sep_len_virg = sep_len_virg.reshape((1, -1))
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
x_vers, y_vers = ecdf(sep_len_vers)
x_virg, y_virg = ecdf(sep_len_virg)

# Generate plot
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none', color = 'green')
_ = plt.plot(x_virg, y_virg, marker='.', linestyle='none', color = 'red')

# Label the axes
_ = plt.xlabel('sepal length (cm)')
_ = plt.ylabel('ECDF')

plt.show()

# =============================================================================
# Generate a permutation sample from two data sets
# =============================================================================
def permutation_sample(data1, data2):
    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

permutation_sample(sep_len_vers, sep_len_virg)

# =============================================================================
# Generating permutation replicates
# =============================================================================
def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates
# =============================================================================
# Diff of means
# =============================================================================
def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""
    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)
    return diff

# Compute difference of mean impact force from experiment: empirical_diff_means
empirical_diff_means = np.mean(sep_len_vers) - np.mean(sep_len_virg)

# Draw 10,000 permutation replicates: perm_replicates
perm_replicates = draw_perm_reps(sep_len_vers, sep_len_virg,
                                 diff_of_means, size=10000)

# =============================================================================
# Vizlualising permutation sampling
# =============================================================================
for _ in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(sep_len_vers, sep_len_virg)

    # Compute ECDFs
    x_1, y_1 = ecdf(perm_sample_1)
    x_2, y_2 = ecdf(perm_sample_2)

    # Plot ECDFs of permutation sample
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)

# Create and plot ECDFs from original data
x_1, y_1 = ecdf(sep_len_vers)
x_2, y_2 = ecdf(sep_len_virg)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('sepal length(cm)')
_ = plt.ylabel('ECDF')
plt.show()


# =============================================================================
# Mean equality testing, compute the p-value
# =============================================================================
nb_samples = 10000

diff_of_means = np.mean(sep_len_virg) - np.mean(sep_len_vers)
boot_diff = np.empty(nb_samples)

for i in range(nb_samples):
    a=np.concatenate((sep_len_vers, sep_len_virg))  
    a=np.random.permutation(a)
    perm_sample_1 = a[:len(sep_len_vers)]
    perm_sample_2 = a[len(sep_len_vers):]
    boot_diff[i] = np.mean(perm_sample_1) - np.mean(perm_sample_2)
    #print(np.mean(perm_sample_1) - np.mean(perm_sample_2))
    #diffs_of_diff[i] = boot_diff - diff_of_means
    
p_value = np.sum(boot_diff >= diff_of_means)/len(boot_diff)
print("end:", diff_of_means, p_value)

_ = plt.hist(boot_diff, bins = 100, density = True)
_ = plt.axvline(x = diff_of_means, label='diff of means', color = 'red')    

plt.show()







