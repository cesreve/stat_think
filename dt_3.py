# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:32:11 2019

@author: wilders
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# =============================================================================
# No ideai what I am doing
# =============================================================================

# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = np.random.poisson(10, size = 10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
n = [20, 100, 1000]
p = [0.5, 0.1, 0.01]

# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n[i], p[i], 10000)
    

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))


# =============================================================================
# Normal distribution
# =============================================================================
# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1 = np.random.normal(20, 1, 100000)
samples_std3 = np.random.normal(20, 3, 100000)
samples_std10 = np.random.normal(20, 10, 100000)

# Make histograms
_ = plt.hist(samples_std1, bins = 100, normed = True, histtype='step')
_ = plt.hist(samples_std3, bins = 100, normed = True, histtype='step')
_ = plt.hist(samples_std10, bins = 100, normed = True, histtype='step')

# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()
plt.close('all')