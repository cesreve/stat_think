# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:25:57 2019

@author: cesreve
"""
# =============================================================================
# libraries
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Data generation
# =============================================================================

np.random.seed(42)

x = np.random.random(size = 1000)
y = 2*x -1.2*x**2 + 1 + np.random.normal(0.7,0.05, size = 1000)
#
#_ = plt.plot(x, y, linestyle='none', marker = '.')
#plt.show()

# =============================================================================
# Linear regression
# =============================================================================

# Plot the illiteracy rate versus fertility
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel("Xs'")
_ = plt.ylabel("Ys'")
plt.margins(0.02)

# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(x, y, 1)

# Print the results to the screen
print('slope =', a)
print('intercept =', b)

# Make theoretical line to plot
x_th = np.array([0, 1])
y_th = a * x_th + b

# Add regression line to your plot
_ = plt.plot(x_th, y_th, color= 'green')

# Perform a degree2 linear of regression using np.polyfit(): a, b, c
a, b, c = np.polyfit(x, y, 2)

# Make theoretical line to plot
x_th2 = np.linspace(0, 1, num=500)
y_th2 = a * x_th2**2 + b * x_th2 + c

# Add regression line to your plot
_ = plt.plot(x_th2, y_th2, color = 'red')

# Draw the plot
plt.show()

# =============================================================================
# Boostrap sampling for coefficient estimation
# =============================================================================
a_s = []
b_s = []

for i in range(10000):
    idx = np.arange(len(x))
    bs_idx = np.random.choice(idx, len(idx))
    bs_x = x[bs_idx]
    bs_y = y[bs_idx]
    a, b = np.polyfit(bs_x, bs_y, 1)
    a_s.append(a)
    b_s.append(b)
    
_ = plt.hist(a_s, bins = 100, density = True, stacked = True)
_ = plt.xlabel('a_s')
_ = plt.ylabel('PDF')
plt.show() 

# =============================================================================
# Plotting bootstraped regression lines
# =============================================================================

for i in range(100):
    x_th = np.array([0, 1])
    y_th = a_s[i] * x_th + b_s[i]
    _ = plt.plot(x_th, y_th, linewidth=0.5, alpha=0.2, color = 'red')
_ = plt.plot(x, y, marker='.', linestyle='none')

plt.show()