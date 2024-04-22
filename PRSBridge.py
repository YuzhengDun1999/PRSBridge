import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from bayesbridge.prsbridge import PrsBridge, RegressionCoefPrior
from bayesbridge.model import PRSModel
import pandas as pd
import argparse
import math
import re
import Data_process

parser = argparse.ArgumentParser()
parser.add_argument('--percent', dest='percent_input', type=float, help='percent')
parser.add_argument('--chr', dest='chr_input', type=int, help='chr')
parser.add_argument('--alpha', dest='alpha_input', type=float, help='alpha')
parser.add_argument('--ref', dest='ref_input', help='reference file location')
parser.add_argument('--sumdat', dest='sumdat_input', help='sumdat file location')
parser.add_argument('--h2', dest='h2', type=float, help='heritability')
parser.add_argument('--h2_se', dest='h2_se', type=float, help='standard error of heritability')
parser.add_argument('--method', dest='method', help='method')
parser.add_argument('--output', dest='output', help='output file location')
args = parser.parse_args()
eigen_rm = args.percent_input
chr = args.chr_input
alpha = args.alpha_input
ref = args.ref_input
sumdat_file = args.sumdat_input
h2 = args.h2
h2_se = args.h2_se
method = args.method
output = args.output

sumdat, SNP_list, A1_list, A2_list = Data_process.parse_sumstats(ref + str(chr), sumdat_file)
n_blk = len(sumdat)
ld_blk = [np.load(ref + str(chr) + '/block' + str(blk)+'.npy') for blk in range(1, n_blk+1)]

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
result_df = pd.DataFrame({'SNP': SNP_list, 'A1': A1_list, 'A2': A2_list, 'BETA': beta_sum})

sqrt_ld_blk = []
inv_ld_blk = []
proj_blk = []; eigenval_blk = []; eigenvec_blk = []
beta_index = 0
beta_init = np.array(beta_sum)

for blk in range(n_blk):
    idx = np.array(np.ix_(sumdat[blk]['index']))
    ld_blk[blk] = ld_blk[blk][np.ix_(idx[0,], idx[0,])]
    ld_blk[blk] = np.copy(ld_blk[blk])
    eigenval, eigenvec = np.linalg.eigh(ld_blk[blk])
    eigenval[eigenval < 10**(-7)] = 0
    percent = int(np.floor(eigenval.shape[0] * eigen_rm))
    start_idx = max(sum(eigenval < 0.01), percent)
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

if h2 > 0 and h2_se > 0:
    tau2 = h2 / beta_sum.shape[0] * math.gamma(1 / alpha) / math.gamma(3 / alpha)
    log10_mean = math.log10(math.sqrt(tau2))
    log10_sd = 0.5 * h2_se * math.log10(math.e) / h2
    gscale_prior = {'log10_mean': log10_mean,'log10_sd': log10_sd}
else:
    gscale_prior = None

model = PRSModel(beta_sum, ld_blk, sqrt_ld_blk, eigenval_blk, eigenvec_blk, np.mean(sample_size), blk_size, 0)

prior = RegressionCoefPrior(
    bridge_exponent=alpha,
    n_fixed_effect=0,
    # Number of coefficients with Gaussian priors of pre-specified sd.
    sd_for_intercept=float('inf'),
    # Set it to float('inf') for a flat prior.
    sd_for_fixed_effect=1.,
    regularizing_slab_size=2,
    global_scale_prior_hyper_param=gscale_prior,
    _global_scale_parametrization='raw'
    # Weakly constrain the magnitude of coefficients under bridge prior.
)

bridge = PrsBridge(model, prior)

mm = 0
beta_init = beta_init * math.sqrt(h2 / np.sum(beta_init**2)) * 10

n_burnin_input = 500
n_iter_input = 800
if alpha == 0.0625:
    n_burnin_input = 1000
    n_iter_input = 1400
if alpha == 0.125:
    n_burnin_input = 600
    n_iter_input = 1000

#n_burnin_input = 0
#n_iter_input = 1000

samples, mcmc_info = bridge.gibbs(
   n_iter=n_iter_input, n_burnin=n_burnin_input, thin=1,
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
result_df['BETA'] = coef
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
if method=='cg':
    np.savetxt(output + '/cg_ite.txt', mcmc_info['_reg_coef_sampling_info']['n_cg_iter'].astype(int), delimiter=',')


