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

#parser = argparse.ArgumentParser()
#parser.add_argument('--percent', dest='percent_input', type=float, help='percent')
#parser.add_argument('--chr', dest='chr_input', type=int, help='chr')
#parser.add_argument('--alpha', dest='alpha_input', type=float, help='alpha')
#parser.add_argument('--race', dest='race_input', help='race')
#parser.add_argument('--ref', dest='ref_input', help='reference file location')
#parser.add_argument('--sumdat', dest='sumdat_input', help='sumdat file location')
#parser.add_argument('--exponent', dest='exponent', type=float, help='exponent of power')
#parser.add_argument('--w1', dest='w1', type=float, help='weight of power')
#parser.add_argument('--h2', dest='h2', type=float, help='heritability')
#parser.add_argument('--h2_se', dest='h2_se', type=float, help='standard error of heritability')
#parser.add_argument('--method', dest='method', help='method')
#parser.add_argument('--output', dest='output', help='output file location')
#args = parser.parse_args()

#eigen_rm = args.percent_input
#chr = args.chr_input
#alpha = args.alpha_input
#Race = args.race_input.split(',')
#ref = args.ref_input.split(',') #'/dcs04/nilanjan/data/ydun/PRS_Bridge/ref_ukbb/chr'
#sumdat_file = args.sumdat_input.split(',')
#exponent = args.exponent
#w1 = args.w1
#h2 = args.h2
#h2_se = args.h2_se
#method = args.method
#output = args.output

eigen_rm = 0.6
chr = 21
alpha = 0.125
Race = 'EUR,AFR'.split(',')
ref = '/Users/yuzhengdun/Documents/Fairness/ref_1kg/EUR/chr,/Users/yuzhengdun/Documents/Fairness/ref_1kg/AFR/chr'.split(',') #'/dcs04/nilanjan/data/ydun/PRS_Bridge/ref_ukbb/chr'
sumdat_file = '/Users/yuzhengdun/Documents/Fairness/EUR/LDL_GLGC/chr21/mega_sumdat.txt,/Users/yuzhengdun/Documents/Fairness/AFR/LDL_GLGC/chr21/mega_sumdat.txt'.split(',')
exponent = 2
w1 = 0.6
h2 = 0.00128
h2_se = 0.000494772675074119
method = 'cg'
output = '/Users/yuzhengdun/Documents/Fairness/EUR/LDL_GLGC/chr21/'
ld_blk = {}; sumdat = {}
n_pop = len(Race)
for i in range(n_pop):
    sumdat[i], SNP_list, A1_list, A2_list = Data_process.parse_sumstats(ref[i] + str(chr), sumdat_file[i])
    n_blk = len(sumdat[i])
    ld_blk[i] = [np.load(ref[i] + str(chr) + '/block' + str(blk) + '.npy') for blk in range(1, n_blk + 1)]

beta_sum = {}; sample_size = {}; scale = {}; idx = {}; se = {}; frq = {}
temp = pd.merge(sumdat[0][0],sumdat[1][0],how='inner',on=['SNP'])
result_df = temp[['SNP', 'A2_y', 'beta_y']]
beta_sum[0] = np.array(temp['new_BETA_x']); beta_sum[1] = np.array(temp['new_BETA_y'])
sample_size[0] = np.array(temp['N_x']); sample_size[1] = np.array(temp['N_y'])
scale[0] = np.array(temp['scale_x']); scale[1] = np.array(temp['scale_y'])
se[0] = np.array(temp['SE_x']); se[1] = np.array(temp['SE_y'])
frq[0] = np.array(temp['frq_x']); frq[1] = np.array(temp['frq_y'])
blk_size = [beta_sum[0].shape[0]]
idx[0] = [[] for _ in range(n_blk)]
idx[1] = [[] for _ in range(n_blk)]
idx[0][0] = temp['index_x']; idx[1][0] = temp['index_y']
for blk in range(1, n_blk):
    temp = pd.merge(sumdat[0][blk], sumdat[1][blk],how='inner',on=['SNP'])
    result_df = pd.concat([result_df, temp[['SNP', 'A2_y', 'beta_y']]])
    beta_sum[0] = np.append(beta_sum[0], np.array(temp['new_BETA_x'])); beta_sum[1] = np.append(beta_sum[1], np.array(temp['new_BETA_y']))
    sample_size[0] = np.append(sample_size[0], np.array(temp['N_x'])); sample_size[1] = np.append(sample_size[1], np.array(temp['N_y']))
    scale[0] = np.append(scale[0], np.array(temp['scale_x'])); scale[1] = np.append(scale[1], np.array(temp['scale_y']))
    se[0] = np.append(se[0], np.array(temp['SE_x'])); se[1] = np.append(se[1], np.array(temp['SE_y']))
    frq[0] = np.append(frq[0], np.array(temp['frq_x'])); frq[1] = np.append(frq[1], np.array(temp['frq_y']))
    blk_size.append(temp.shape[0])
    idx[0][blk] = temp['index_x']; idx[1][blk] = temp['index_y']

for i in range(n_pop):
    beta_sum[i] = beta_sum[i].astype(float)
    sample_size[i] = sample_size[i].astype(float)
    scale[i] = scale[i].astype(float)
Omega = {}
weight_0 = (frq[0] * (1-frq[0])) ** exponent
weight_1 = (frq[1] * (1-frq[1])) ** exponent
weight = w1 * weight_0 / (w1 * weight_0 + (1 - w1) * weight_1)
weight[weight_0 > weight_1] = w1
Omega[0] = weight / se[0]**2 / (weight/se[0]**2 + (1-weight)/se[1]**2) * scale[0]
Omega[1] = (1-weight) / se[1]**2 / (weight/se[0]**2 + (1-weight)/se[1]**2) * scale[1]

#Omega[0] = w1 / se[0]**2 / (w1/se[0]**2 + (1-w1)/se[1]**2) * scale[0]
#Omega[1] = (1-w1) / se[1]**2 / (w1/se[0]**2 + (1-w1)/se[1]**2) * scale[1]
beta_meta = Omega[0] * beta_sum[0] + Omega[1] * beta_sum[1]

result_df = result_df.rename(columns={"A2_y": "ALT", "beta_y": "BETA"})

ld_blk_all = []; sqrt_ld_blk = []; inv_ld_blk = []; proj_blk = []; eigenval_blk = []; eigenvec_blk = []
beta_sum_all = beta_meta.copy(); beta_init = beta_meta.copy()
beta_index = 0
for blk in range(n_blk):
    for i in range(n_pop):
        idx_tmp = np.array(np.ix_(idx[i][blk]))
        ld_blk[i][blk] = ld_blk[i][blk][np.ix_(idx_tmp[0,], idx_tmp[0,])]
    idx_blk = range(beta_index, beta_index + blk_size[blk])
    M1_tmp = np.diag(Omega[0][idx_blk]).dot(ld_blk[0][blk]) + \
             np.diag(Omega[1][idx_blk]).dot(ld_blk[1][blk])
    M2_tmp = np.diag(Omega[0][idx_blk]).dot(ld_blk[0][blk]).dot(np.diag(Omega[0][idx_blk]))/np.mean(sample_size[0]) + \
             np.diag(Omega[1][idx_blk]).dot(ld_blk[1][blk]).dot(np.diag(Omega[1][idx_blk]))/np.mean(sample_size[1])
    M2_eigenval, M2_eigenvec = np.linalg.eigh(M2_tmp)
    percent = int(np.floor(M2_eigenval.shape[0] * eigen_rm))
    start_idx = max(sum(M2_eigenval < 0.01 / (w1 * np.mean(sample_size[0]) + (1-w1) * np.mean(sample_size[1]))), percent)
    M2_eigenval_new = np.copy(M2_eigenval[start_idx:]); M2_eigenvec_new = np.copy(M2_eigenvec[:, start_idx:])
    inv_eigenval = 1 / M2_eigenval_new
    #inv_eigenval[M2_eigenval_new > 0] = 1 / inv_eigenval[inv_eigenval > 0]
    beta_sum_all[idx_blk] = M1_tmp.T.dot(M2_eigenvec_new).dot(np.diag(inv_eigenval)).dot(M2_eigenvec_new.T).\
        dot(M2_eigenvec_new).dot(M2_eigenvec_new.T).dot(beta_meta[idx_blk])
    ld_tmp = M1_tmp.T.dot(M2_eigenvec_new).dot(np.diag(inv_eigenval)).dot(M2_eigenvec_new.T).dot(M1_tmp)
    ld_eigenval, ld_eigenvec = np.linalg.eigh(ld_tmp)
    ld_eigenval_new = np.copy(ld_eigenval[start_idx:]); ld_eigenvec_new = np.copy(ld_eigenvec[:, start_idx:])
    sqrt_ld_blk.append(ld_eigenvec_new.dot(np.diag(np.sqrt(ld_eigenval_new))).dot(ld_eigenvec_new.T))
    ld_blk_all.append(ld_tmp)
    eigenval_blk.append(ld_eigenval_new)
    eigenvec_blk.append(ld_eigenvec_new)
    beta_index = beta_index + ld_blk[0][blk].shape[0]


if h2 > 0 and h2_se > 0:
    tau2 = h2 / beta_sum_all.shape[0] * math.gamma(1 / alpha) / math.gamma(3 / alpha) * np.mean(scale[0]**2*w1 + scale[1]**2*(1-w1))
    log10_mean = math.log10(math.sqrt(tau2))
    log10_sd = 0.5 * h2_se * math.log10(math.e) / h2 * np.sqrt(np.mean(scale[0]**2*w1 + scale[1]**2*(1-w1)))
    gscale_prior = {'log10_mean': log10_mean,'log10_sd': log10_sd}
else:
    gscale_prior = None

model = PRSModel(beta_sum_all, ld_blk_all, sqrt_ld_blk, inv_ld_blk, proj_blk, eigenval_blk, eigenvec_blk, 1, blk_size, 0)
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
result_df.to_csv(output + '/coef_' + str(w1)+ '_' + str(exponent) + '.txt', sep='\t', index=False)
