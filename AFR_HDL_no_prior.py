import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from bayesbridge.prsbridge import PrsBridge, RegressionCoefPrior
from bayesbridge.model import PRSModel
import pandas as pd
import argparse
import math
import re

parser = argparse.ArgumentParser()
parser.add_argument('--percent', dest='percent_input', type=float, help='percent')
parser.add_argument('--chr', dest='chr_input', type=int, help='chr')
parser.add_argument('--alpha', dest='alpha_input', type=float, help='alpha')
parser.add_argument('--ref', dest='ref_input', help='reference file location')
parser.add_argument('--hfile', dest='heritability', help='hertability file location')
parser.add_argument('--sumdat', dest='sumdat_input', help='sumdat file location')
parser.add_argument('--method', dest='method', help='method')
parser.add_argument('--output', dest='output', help='output file location')
args = parser.parse_args()
eigen_rm = args.percent_input
chr = args.chr_input
alpha =args.alpha_input
ref = args.ref_input #'/dcs04/nilanjan/data/ydun/PRS_Bridge/ref_ukbb/chr'
sumdat_file = args.sumdat_input #'/dcs04/nilanjan/data/ydun/PRS_Bridge/data_Rcov/chr'
hfile = args.heritability
method = args.method
output = args.output

ldblk_dir = ref + str(chr)
n_blk = np.loadtxt(ref + str(chr) + '/block.txt').astype(int)
n_blk = np.max(n_blk)
ldsc_file = hfile + '_h2.txt'

ld_blk = [np.load(ldblk_dir+'/block'+str(blk)+'.npy') for blk in range(1,n_blk+1)]
sumdat = [pd.read_csv(sumdat_file+str(blk)+'.txt') for blk in range(1,n_blk+1)]
result_df = sumdat[0][['SNP_ID', 'ALT', 'BETA']]
beta_sum = np.array(sumdat[0]['new_BETA'])
sample_size = np.array(sumdat[0]['N'])
blk_size = [beta_sum.shape[0]]
scale = np.array(sumdat[0]['scale'])
for blk in range(1,n_blk):
    beta_sum = np.append(beta_sum, np.array(sumdat[blk]['new_BETA']))
    sample_size = np.append(sample_size, np.array(sumdat[blk]['N']))
    blk_size.append(sumdat[blk].shape[0])
    scale = np.append(scale, np.array(sumdat[blk]['scale']))
    result_df = pd.concat([result_df, sumdat[blk][['SNP_ID', 'ALT', 'BETA']]])

beta_sum = beta_sum.astype(float); sample_size = sample_size.astype(float); scale = scale.astype(float)

sqrt_ld_blk = []
inv_ld_blk = []
proj_blk = []; eigenval_blk = []; eigenvec_blk = []
beta_index = 0
eigen_rm = 0.5

for blk in range(n_blk):
    idx = np.array(np.ix_(sumdat[blk]['index']))-1
    ld_blk[blk] = ld_blk[blk][np.ix_(idx[0,], idx[0,])]
    ld_blk[blk] = ld_blk[blk] #+ np.diag(np.ones(ld_blk[blk].shape[0]))
    eigenval, eigenvec = np.linalg.eigh(ld_blk[blk])
    eigenval[eigenval < 10 ** (-7)] = 0
    percent = int(np.floor(eigenval.shape[0] * eigen_rm))
    start_idx = max(sum(eigenval < 0.05), percent)
    ld_blk[blk] = eigenvec[:, start_idx:].dot(np.diag(eigenval[start_idx:])).dot(eigenvec[:, start_idx:].T)
    sqrt_ld_blk.append(eigenvec[:, start_idx:].dot(np.diag(np.sqrt(eigenval[start_idx:]))).dot(eigenvec[:, start_idx:].T))
    inv_eigenval = eigenval.copy()
    inv_eigenval[inv_eigenval > 0] = 1 / inv_eigenval[inv_eigenval > 0]
    inv_ld_blk.append(eigenvec[:, start_idx:].dot(np.diag(inv_eigenval[start_idx:])).dot(eigenvec[:, start_idx:].T))
    proj_blk.append(eigenvec[:, start_idx:].dot(eigenvec[:, start_idx:].T))
    eigenval_blk.append(np.copy(eigenval[start_idx:]))
    eigenvec_blk.append(np.copy(eigenvec[:, start_idx:]))
    beta_sum[beta_index: (beta_index+eigenval.shape[0])] = eigenvec[:, start_idx:].dot(eigenvec[:, start_idx:].T).dot(beta_sum[beta_index: (beta_index+eigenval.shape[0])])
    beta_index = beta_index + eigenval.shape[0]

h2=0.01
h2_se=0.0001
tau2 = h2/beta_sum.shape[0] * math.gamma(1/alpha) / math.gamma(3/alpha)
log10_mean = math.log10(math.sqrt(tau2))
log10_sd = 0.5*h2_se*math.log10(math.e)/h2

model = PRSModel(beta_sum, ld_blk, sqrt_ld_blk, inv_ld_blk, proj_blk, eigenval_blk, eigenvec_blk, np.mean(sample_size), blk_size, 0)

prior = RegressionCoefPrior(
    bridge_exponent=alpha,
    n_fixed_effect=0,
    #shape=shape,
    #rate=rate,
    # Number of coefficients with Gaussian priors of pre-specified sd.
    sd_for_intercept=float('inf'),
    # Set it to float('inf') for a flat prior.
    sd_for_fixed_effect=1.,
    regularizing_slab_size=2,
    #global_scale_prior_hyper_param = {'log10_mean': log10_mean,'log10_sd': log10_sd},
    _global_scale_parametrization='raw'
    # Weakly constrain the magnitude of coefficients under bridge prior.
)

bridge = PrsBridge(model, prior)

mm = 0
beta_init = np.array(beta_sum)
for kk in range(n_blk):
    if blk_size[kk] == 0:
        continue
    else:
        idx_blk = range(mm, mm + blk_size[kk])
        beta_init[idx_blk] = inv_ld_blk[kk].dot(np.array(beta_sum)[idx_blk])
        mm += blk_size[kk]
beta_init = beta_init * math.sqrt( h2 / np.sum(beta_init**2)) * 10

samples, mcmc_info = bridge.gibbs(
    n_iter=800, n_burnin=500, thin=1,
    #init={'global_scale': 0.01, 'coef': beta_init,'local_scale': lscale_init},
    #init={'global_scale': 0.01},
    init={'coef': beta_init},
    params_to_save=(('coef', 'global_scale')),
    coef_sampler_type=method,
    seed=111, max_iter=2000
)
coef_samples = samples['coef']  # Extract all but the intercept
coef = np.mean(coef_samples, axis=1)
coef = coef * scale
#np.savetxt(output + '/coef.txt', coef.astype(float), delimiter=',')
result_df['BETA'] = coef[0,:]
result_df.to_csv(output + '/coef.txt', sep='\t', index=False)

plt.figure(figsize=(12, 5))
plt.rcParams['font.size'] = 20

plt.cla()
plt.plot(samples['global_scale'])
plt.savefig(output + '/samples_gscale.pdf')
plt.cla()
plt.semilogy(samples['global_scale'])
plt.savefig(output + '/samples_log_gscale.pdf')
plt.cla()

time = pd.DataFrame({'time':[mcmc_info['runtime']]})
print(mcmc_info['runtime'])
time.to_csv(output + '/time.txt', index=0, header=0)
np.savetxt(output + '/cg_ite.txt', mcmc_info['_reg_coef_sampling_info']['n_cg_iter'].astype(int), delimiter=',')

