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
parser.add_argument('--race', dest='race_input', help='race')
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
Race = args.race_input.split(',')
ref = args.ref_input.split(',') #'/dcs04/nilanjan/data/ydun/PRS_Bridge/ref_ukbb/chr'
sumdat_file = args.sumdat_input.split(',')
h2 = args.h2
h2_se = args.h2_se
method = args.method
output = args.output
ld_blk = {}; sumdat = {}
n_pop = len(Race)
for i in range(n_pop):
    sumdat[i], SNP_list, A1_list, A2_list = Data_process.parse_sumstats(ref[i] + str(chr), sumdat_file[i])
    n_blk = len(sumdat[i])
    ld_blk[i] = [np.load(ref[i] + str(chr) + '/block' + str(blk) + '.npy') for blk in range(1, n_blk + 1)]

beta_sum = {}; sample_size = {}; scale = {}; idx = {}
temp = pd.merge(sumdat[0][0],sumdat[1][0],how='inner',on=['SNP'])
result_df = temp[['SNP', 'A2_y', 'beta_y']]
beta_sum[0] = np.array(temp['new_BETA_x']); beta_sum[1] = np.array(temp['new_BETA_y'])
sample_size[0] = np.array(temp['N_x']); sample_size[1] = np.array(temp['N_y'])
scale[0] = np.array(temp['scale_x']); scale[1] = np.array(temp['scale_y'])
blk_size = [beta_sum[0].shape[0]]
idx[0] = [[] for _ in range(n_blk)]
idx[1] = [[] for _ in range(n_blk)]
idx[0][0] = temp['index_x']; idx[1][0] = temp['index_y']
for blk in range(1, n_blk):
    temp = pd.merge(sumdat[0][blk],sumdat[1][blk],how='inner',on=['SNP'])
    result_df = pd.concat([result_df, temp[['SNP', 'A2_y', 'beta_y']]])
    beta_sum[0] = np.append(beta_sum[0], np.array(temp['new_BETA_x']));beta_sum[1] = np.append(beta_sum[1], np.array(temp['new_BETA_y']))
    sample_size[0] = np.append(sample_size[0], np.array(temp['N_x']));sample_size[1] = np.append(sample_size[1], np.array(temp['N_y']))
    scale[0] = np.append(scale[0], np.array(temp['scale_x']));scale[1] = np.append(scale[1], np.array(temp['scale_y']))
    blk_size.append(temp.shape[0])
    idx[0][blk] = temp['index_x']; idx[1][blk] = temp['index_y']

result_df = result_df.rename(columns={"A2_y": "ALT", "beta_y": "BETA"})
for i in range(n_pop):
    beta_sum[i] = beta_sum[i].astype(float)
    sample_size[i] = sample_size[i].astype(float)
    scale[i] = scale[i].astype(float)

sample_size_all = sample_size[0] + sample_size[1]
beta_sum_all = (beta_sum[0] * sample_size[0] + beta_sum[1] * sample_size[1]) / sample_size_all
ld_blk_all = []; sqrt_ld_blk = []; inv_ld_blk = []; proj_blk = []; eigenval_blk = []; eigenvec_blk = []

for blk in range(n_blk):
    for i in range(n_pop):
        idx_tmp = np.array(np.ix_(idx[i][blk]))
        ld_blk[i][blk] = ld_blk[i][blk][np.ix_(idx_tmp[0,], idx_tmp[0,])]
    ld_blk_all.append((ld_blk[0][blk] * np.mean(sample_size[0]) + ld_blk[1][blk] * np.mean(sample_size[1])) / np.mean(sample_size_all))

beta_index = 0
beta_init = np.array(beta_sum_all)
for blk in range(n_blk):
    eigenval, eigenvec = np.linalg.eigh(ld_blk_all[blk])
    eigenval[eigenval < 10**(-7)] = 0
    percent = int(np.floor(eigenval.shape[0] * eigen_rm))
    start_idx = max(sum(eigenval < 0.005), percent)#previous results are 0.05
    eigenval_new = np.copy(eigenval[start_idx:]); eigenvec_new = np.copy(eigenvec[:, start_idx:])
    ld_blk_all[blk] = eigenvec_new.dot(np.diag(eigenval_new)).dot(eigenvec_new.T)
    sqrt_ld_blk.append(eigenvec_new.dot(np.diag(np.sqrt(eigenval_new))).dot(eigenvec_new.T))
    inv_eigenval = eigenval.copy()
    inv_eigenval[inv_eigenval > 0] = 1 / inv_eigenval[inv_eigenval > 0]
    eigenval_blk.append(eigenval_new)
    eigenvec_blk.append(eigenvec_new)
    beta_sum_all[beta_index: (beta_index+eigenval.shape[0])] = eigenvec[:, start_idx:].dot(eigenvec[:, start_idx:].T).dot(beta_sum_all[beta_index: (beta_index+eigenval.shape[0])])
    beta_init[beta_index: (beta_index+eigenval.shape[0])] = eigenvec_new.dot(np.diag(inv_eigenval[start_idx:])).dot(eigenvec_new.T).dot(np.array(beta_sum_all)[beta_index: (beta_index+eigenval.shape[0])])
    beta_index = beta_index + eigenval.shape[0]

if h2 > 0 and h2_se > 0:
    tau2 = h2 / beta_sum_all.shape[0] * math.gamma(1 / alpha) / math.gamma(3 / alpha)
    log10_mean = math.log10(math.sqrt(tau2))
    log10_sd = 0.5 * h2_se * math.log10(math.e) / h2
    gscale_prior = {'log10_mean': log10_mean,'log10_sd': log10_sd}
else:
    gscale_prior = None

model = PRSModel(beta_sum_all, ld_blk_all, sqrt_ld_blk, inv_ld_blk, proj_blk, eigenval_blk, eigenvec_blk, np.mean(sample_size_all), blk_size, 0)
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
    global_scale_prior_hyper_param=gscale_prior,
    _global_scale_parametrization='raw'
    # Weakly constrain the magnitude of coefficients under bridge prior.
)

bridge = PrsBridge(model, prior)
beta_init = beta_init * math.sqrt(h2 / np.sum(beta_init**2)) * 10
n_brunin_input = 500
n_iter_input = 800
if alpha == 0.0625:
    n_brunin_input = 1100
    n_iter_input = 1500
if alpha == 0.125:
    n_brunin_input = 600
    n_iter_input = 1000

samples, mcmc_info = bridge.gibbs(
    n_iter=n_iter_input, n_burnin=n_brunin_input, thin=1,
    #init={'global_scale': 0.01, 'coef': beta_init,'local_scale': lscale_init},
    #init={'global_scale': 0.01},
    init={'coef': beta_init},
    params_to_save=(('coef', 'global_scale')),
    coef_sampler_type='cg',
    seed=111, max_iter=2000
)
coef_samples = samples['coef']  # Extract all but the intercept
coef = np.mean(coef_samples, axis=1)
result_df['BETA'] = coef
result_df.to_csv(output + '/coef_standardized.txt', sep='\t', index=False)

coef = np.array(pd.read_csv(output + '/coef_standardized.txt', sep='\t')['BETA'])
coef_new = coef.copy()
Coef_Percentile = np.array([99.99, 99.95, 99.9, 99.5, 99, 98, 97, 96, 95, 90])
Lambda1 = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3])
Lambda2 = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4])
for lambda1 in Lambda1:
    for lambda2 in Lambda2:
        for coef_percentile in Coef_Percentile:
            mm = 0
            coef_threshold = np.percentile(np.abs(coef), coef_percentile)
            for kk in range(n_blk):
                if blk_size[kk] == 0:
                    continue
                else:
                    idx_blk = range(mm, mm + blk_size[kk])
                    coef_idx = coef[idx_blk]
                    coef_idx_pos = np.abs(coef_idx)
                    idx_threshold = np.where(coef_idx_pos >= coef_threshold)[0]
                    mm += blk_size[kk]
                    if idx_threshold.shape == 0:
                        continue
                    else:
                        temp = coef_new[idx_blk].copy()
                        temp[idx_threshold] = np.linalg.solve(
                            np.identity(idx_threshold.shape[0]) + lambda1 * ld_blk[0][kk][
                                np.ix_(idx_threshold, idx_threshold)] + lambda2 * ld_blk[1][kk][
                                np.ix_(idx_threshold, idx_threshold)],
                            coef[idx_blk][idx_threshold] + lambda1 * beta_sum[0][idx_blk][idx_threshold] + lambda2 *
                            beta_sum[1][idx_blk][idx_threshold])
                        coef_new[idx_blk] = temp.copy()
            result_df['BETA'] = coef_new * scale[0]
            result_df.to_csv(''.join([output, '/coef_EUR_percent_', str(coef_percentile), '_lambda1_' + str(lambda1) + '_lambda2_', str(lambda2), '.txt']),
                             sep='\t', index=False)
            result_df['BETA'] = coef_new * scale[1]
            result_df.to_csv(''.join([output, '/coef_AFR_percent_', str(coef_percentile), '_lambda1_' + str(lambda1) + '_lambda2_', str(lambda2), '.txt']),
                             sep='\t', index=False)