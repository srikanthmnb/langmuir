# -*- coding: utf-8 -*-
"""
This script fits Langmuir adsorption isotherm to experimental adsorption data
"""
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model

adsorbate_bulk = [0.00025,0.0025,0.5,2.5,25]
scale_factor = 50
coverage = [scale_factor*x for x in [0.00031, 0.00496, 0.01639, 0.0187, 0.01601]]
coverage_error = [scale_factor*x for x in [1.61e-5, 1.06e-4, 3.98e-4, 4.68e-4, 1.03e-3]]

def langmuir(c,K):
    return np.divide(K*c, (1+K*c))

lmodel = Model(langmuir)
result = lmodel.fit(coverage, c = adsorbate_bulk, K = 1,  weights = np.divide(1,coverage_error))
print(result.fit_report())

concentrations_fit = np.logspace(-5,2, 101)
coverage_fit = langmuir(concentrations_fit, result.best_values['K'])

[fig,ax] = plt.subplots(1,1, figsize = [6,6])
ax.errorbar(adsorbate_bulk, coverage,  coverage_error, None, 'ko', mfc = "None")
ax.plot(concentrations_fit, coverage_fit, 'k-')
ax.set_xscale('log')
ax.set_ylabel(r'Coverage, $\theta$', fontsize=12)
ax.set_xlabel('mM of [SeCN$^{-}$]', fontsize=12)
ax.text(0.04,0.35, r'$ \theta = \frac{Kc}{1+Kc}$', fontsize = 16)
ax.text(1e-2,0.2,'$K = 101 \pm 20$, mM$^{-1}$', fontsize = 12)
fig.savefig('langmuir_xfntr_50.jpg', bbox_inches = 'tight')