import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from bayesbridge.prsbridge import PrsBridge, RegressionCoefPrior
from bayesbridge.model import PRSModel
import pandas as pd
import argparse
import math
import re

ldblk_dir = '../../MultiEthnic/AFR/ref_hapmap3/chr22'
n_blk = 10
eigen_rm = 0.5
sumdat_file = '../../MultiEthnic/AFR/sumdat/chr22/sumdat-1kg-block'
ldsc_file = '../sumdat/chr22/chr22_h2.txt'
alpha = 0.25
ld_blk = [np.load(ldblk_dir+'/block'+str(blk)+'.npy') for blk in range(1,n_blk+1)]
sumdat = [pd.read_csv(sumdat_file+str(blk)+'.txt') for blk in range(1,n_blk+1)]

beta_sum = np.array(sumdat[0]['new_BETA'])
sample_size = np.array(sumdat[0]['N'])
blk_size = [beta_sum.shape[0]]
scale = np.array(sumdat[0]['scale'])
for blk in range(1,n_blk):
    beta_sum = np.append(beta_sum, np.array(sumdat[blk]['new_BETA']))
    sample_size = np.append(sample_size, np.array(sumdat[blk]['N']))
    blk_size.append(sumdat[blk].shape[0])
    scale = np.append(scale, np.array(sumdat[blk]['scale']))
beta_sum = beta_sum.astype(float); sample_size = sample_size.astype(float); scale = scale.astype(float)

sqrt_ld_blk = []; inv_ld_blk = []; proj_blk = []; eigenval_blk = []; eigenvec_blk = []
beta_index = 0

beta_init = np.array(beta_sum)

for blk in range(n_blk):
    idx = np.array(np.ix_(sumdat[blk]['index']))-1
    ld_blk[blk] = ld_blk[blk][np.ix_(idx[0,], idx[0,])]
    ld_blk[blk] = np.copy(ld_blk[blk]) #+ np.diag(np.ones(ld_blk[blk].shape[0])) * 10**(-5)
    eigenval, eigenvec = np.linalg.eigh(ld_blk[blk])
    eigenval[eigenval < 10**(-7)] = 0
    percent = int(np.floor(eigenval.shape[0] * eigen_rm))
    start_idx = max(sum(eigenval < 0.005), percent)#previous results are 0.05
    eigenval_new = np.copy(eigenval[start_idx:]); eigenvec_new = np.copy(eigenvec[:, start_idx:])
    ld_blk[blk] = eigenvec_new.dot(np.diag(eigenval_new)).dot(eigenvec_new.T)
    sqrt_ld_blk.append(eigenvec_new.dot(np.diag(np.sqrt(eigenval_new))).dot(eigenvec_new.T))
    inv_eigenval = eigenval.copy()
    inv_eigenval[inv_eigenval > 0] = 1 / inv_eigenval[inv_eigenval > 0]
    eigenval_blk.append(eigenval_new)
    eigenvec_blk.append(eigenvec_new)
    beta_sum[beta_index: (beta_index+eigenval.shape[0])] = eigenvec[:, start_idx:].dot(eigenvec[:, start_idx:].T).dot(beta_sum[beta_index: (beta_index+eigenval.shape[0])])
    beta_init[beta_index: (beta_index+eigenval.shape[0])] = eigenvec_new.dot(np.diag(inv_eigenval[start_idx:])).dot(eigenvec_new.T).dot(np.array(beta_sum)[beta_index: (beta_index+eigenval.shape[0])])
    beta_index = beta_index + eigenval.shape[0]

h2 = 0.02; h2_se = 0.002

tau2 = h2/beta_sum.shape[0] * math.gamma(1/alpha) / math.gamma(3/alpha)
log10_mean = math.log10(math.sqrt(tau2))
log10_sd = 0.5*h2_se*math.log10(math.e)/h2
model = PRSModel(beta_sum, ld_blk, sqrt_ld_blk, inv_ld_blk, proj_blk, eigenval_blk, eigenvec_blk, np.mean(sample_size), blk_size, h2)
prior = RegressionCoefPrior(
    bridge_exponent=alpha,
    n_fixed_effect=0,
    # Number of coefficients with Gaussian priors of pre-specified sd.
    sd_for_intercept=float('inf'),
    # Set it to float('inf') for a flat prior.
    sd_for_fixed_effect=1.,
    regularizing_slab_size=2,
    #global_scale_prior_hyper_param={'log10_mean': log10_mean,'log10_sd': log10_sd},
    _global_scale_parametrization='raw'
    # Weakly constrain the magnitude of coefficients under bridge prior.
)

bridge = PrsBridge(model, prior)

M = 2000
I = 100
Lscale = np.ones(M)

N = 1
tau2 = 1/N
alpha = 1
for m in range(M):
    coef1 = np.random.randn(1)
    #gscale = np.sqrt(tau2)
    for i in range(I):
        #gscale = bridge.update_global_scale(
        #    gscale, coef1, alpha,
        #    coef_expected_magnitude_lower_bd=0
        #)
        gscale = np.sqrt(tau2)
        lscale = bridge.update_local_scale(
            gscale, coef1, alpha)
        coef1 = np.random.randn(1) * abs(gscale * lscale)
        Lscale[m] = lscale * gscale

Lscale = np.ones(M)
alpha = 0.125
tau2 = math.gamma(3)/math.gamma(1)/(math.gamma(3/alpha)/math.gamma(1/alpha))
tau2 = 1
for m in range(M):
    coef1 = np.random.randn(1)
    #gscale = np.sqrt(tau2)
    for i in range(I):
        #gscale = bridge.update_global_scale(
        #    gscale, coef1, alpha,
        #    coef_expected_magnitude_lower_bd=0
        #)
        gscale = np.sqrt(tau2)
        lscale = bridge.update_local_scale(
            gscale, coef1, alpha)
        coef1 = np.random.randn(1) * abs(gscale * lscale)
        Lscale[m] = lscale * gscale

plt.figure(figsize=(12, 5))
plt.cla()
temp=1/(1+Lscale**2)
plt.hist(temp)
plt.title("alpha=" + str(alpha))
plt.show()

alpha = 0.75
tau = np.sqrt(1/3.35954)

alpha = 0.5
tau = np.sqrt(1/60)

alpha = 0.25
tau = np.sqrt(1/6652800)

for m in range(M):
    coef1 = np.random.randn(1)
    for i in range(I):
        gscale = bridge.update_global_scale(
            gscale, coef1, alpha,
            coef_expected_magnitude_lower_bd=0
        )
        lscale = bridge.update_local_scale(
            gscale, coef1, alpha)
        coef1 = np.random.randn(1) * abs(gscale * lscale)
        Lscale[m] = lscale * gscale

plt.figure(figsize=(12, 5))
plt.cla()
temp=1/(1+10000*Lscale**2)
plt.hist(temp)
plt.title("alpha=" + str(alpha))
plt.show()