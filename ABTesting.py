# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:37:50 2019

@author: cesreve
"""
# =============================================================================
# import librairies
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# AB-Testing
# =============================================================================

# Construct arrays of data: dems, reps
dems = np.array([True] * 153 + [False] * 91)
reps = np.array([True] * 136 + [False] * 35)

def frac_yea_dems(dems, reps):
    """Compute fraction of Democrat yea votes."""
    frac = np.sum(dems) / len(dems)
    return frac

# Acquire permutation samples: perm_replicates
perm_replicates = draw_perm_reps(dems, reps, frac_yea_dems, 10000)

# Compute and print p-value: p
p = np.sum(perm_replicates <= 153/244) / len(perm_replicates)
print('p-value =', p)


# =============================================================================
# Data Generation
# =============================================================================
white = np.array([True] * 153 + [False] * 91)
black = np.array([True] * 136 + [False] * 35)

nb_simul = 10000

true_frac_wh = np.sum(white)/len(white) # true fractions of trues for Whites
true_frac_bk = np.sum(black)/len(black) # true fractions of trues for Whites

sim_frac_wh = np.empty(nb_simul)
sim_frac_bk = np.empty(nb_simul)

for i in range (nb_simul):
    full = np.concatenate((white, black)) # hypothese nulle: tout vient du même échantilllon
    full = np.random.permutation(full) # on remélange
    # on recrée les echantillons
    perm_wh = full[:len(white)] 
    perm_bk = full[len(white):]
    
    #calcul des nouvelles fractions (notre statistiques de test, i.e ce qu'on cherche à tester)
    frac_wh = np.sum(perm_wh)/len(perm_wh) 
    frac_bk = np.sum(perm_bk)/len(perm_bk)
    
    # ajout à la liste des nouvelles fractions 
    sim_frac_wh[i] = frac_wh 
    sim_frac_bk[i] = frac_bk

_ = plt.hist(sim_frac_bk, bins = 100, density = True)
_ = plt.axvline(x = true_frac_bk, label='emprirical fractions of trues for black', color = 'red') 
_ = plt.xlabel('simulated fractions')
_ = plt.ylabel('probability')
plt.show()


print('true fractions', true_frac_wh, true_frac_bk)


_ = plt.hist(sim_frac_wh, bins = 100, alpha = 0.5, density = True)
_ = plt.hist(sim_frac_bk, bins = 100,  alpha = 0.5, density = True)
plt.show()