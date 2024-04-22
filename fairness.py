import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from bayesbridge.prsbridge import PrsBridge, RegressionCoefPrior
from bayesbridge.model import PRSModel
import pandas as pd
import argparse
import math
from scipy.optimize import minimize
import re
'''
parser = argparse.ArgumentParser()
parser.add_argument('--percent', dest='percent_input', type=float, help='percent')
parser.add_argument('--chr', dest='chr_input', type=int, help='chr')
parser.add_argument('--alpha', dest='alpha_input', type=float, help='alpha')
parser.add_argument('--output', dest='output', help='output file location')
args = parser.parse_args()
eigen_rm = args.percent_input
chr = args.chr_input
alpha = args.alpha_input
output = args.output

ldblk_dir = '/dcs04/nilanjan/data/ydun/PRS_Bridge/ref_ukbb/chr' + str(chr)
n_blk = np.loadtxt('/dcs04/nilanjan/data/ydun/PRS_Bridge/ref_ukbb/chr' + str(chr) + '/block.txt').astype(int)
n_blk = np.max(n_blk)
ldsc_file = '/dcs04/nilanjan/data/ydun/PRS_Bridge/data/chr' + str(chr) + '/chr' + str(chr) + '_h2.txt'
sumdat_file = '/dcs04/nilanjan/data/ydun/PRS_Bridge/data/chr' + str(chr) + '/sumdat-ukb-block'
'''

ld_blk = {}; sumdat = {}
Race = ['EUR', 'AFR']
n_pop = len(Race)
n_blk = 4
eigen_rm = 0.5
alpha = 0.125
for i in range(n_pop):
    ldblk_dir = '../../MultiEthnic/' + Race[i] + '/ref_hapmap3/chr22'
    sumdat_file = '../../MultiEthnic/'+ Race[i] + '/sumdat/chr22/sumdat-1kg-block'
    ld_blk[i] = [np.load(ldblk_dir + '/block' + str(blk) + '.npy') for blk in range(1, n_blk + 1)]
    sumdat[i] = [pd.read_csv(sumdat_file + str(blk)+'.txt') for blk in range(1,n_blk+1)]

beta_sum = {}; sample_size = {}; scale = {}; idx = {}
temp = pd.merge(sumdat[0][0],sumdat[1][0],how='inner',on=['SNP_ID'])
result_df = temp[['SNP_ID', 'ALT_y', 'BETA_y']]
beta_sum[0] = np.array(temp['new_BETA_x']); beta_sum[1] = np.array(temp['new_BETA_y'])
sample_size[0] = np.array(temp['N_x']); sample_size[1] = np.array(temp['N_y'])
scale[0] = np.array(temp['scale_x']); scale[1] = np.array(temp['scale_y'])
blk_size = [beta_sum[0].shape[0]]
idx[0] = [[] for _ in range(n_blk)]
idx[1] = [[] for _ in range(n_blk)]
idx[0][0] = temp['index_x']; idx[1][0] = temp['index_y']
for blk in range(1, n_blk):
    temp = pd.merge(sumdat[0][blk],sumdat[1][blk],how='inner',on=['SNP_ID'])
    result_df = pd.concat([result_df, temp[['SNP_ID', 'ALT_y', 'BETA_y']]])
    beta_sum[0] = np.append(beta_sum[0], np.array(temp['new_BETA_x']));beta_sum[1] = np.append(beta_sum[1], np.array(temp['new_BETA_y']))
    sample_size[0] = np.append(sample_size[0], np.array(temp['N_x']));sample_size[1] = np.append(sample_size[1], np.array(temp['N_y']))
    scale[0] = np.append(scale[0], np.array(temp['scale_x']));scale[1] = np.append(scale[1], np.array(temp['scale_y']))
    blk_size.append(temp.shape[0])
    idx[0][blk] = temp['index_x']; idx[1][blk] = temp['index_y']

result_df = result_df.rename(columns={"ALT_y": "ALT", "BETA_y": "BETA"})
for i in range(n_pop):
    beta_sum[i] = beta_sum[i].astype(float)
    sample_size[i] = sample_size[i].astype(float)
    scale[i] = scale[i].astype(float)

sample_size_all = sample_size[0] + sample_size[1]
beta_sum_all = (beta_sum[0] * sample_size[0] + beta_sum[1] * sample_size[1]) / sample_size_all
ld_blk_all = []; sqrt_ld_blk = []; inv_ld_blk = []; proj_blk = []; eigenval_blk = []; eigenvec_blk = []

for blk in range(n_blk):
    for i in range(n_pop):
        idx_tmp = np.array(np.ix_(idx[i][blk])) - 1
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

'''
f = open(ldsc_file, 'r')
temp = f.read()
temp = re.findall(r'\d+\.?\d*', temp.split(':')[1])

h2 = float(temp[0])
h2_se = float(temp[1])
'''
h2 = 0.02; h2_se = 0.002

tau2 = h2/beta_sum_all.shape[0] * math.gamma(1/alpha) / math.gamma(3/alpha)
log10_mean = math.log10(math.sqrt(tau2))
log10_sd = 0.5*h2_se*math.log10(math.e)/h2
model = PRSModel(beta_sum_all, ld_blk_all, sqrt_ld_blk, inv_ld_blk, proj_blk, eigenval_blk, eigenvec_blk, np.mean(sample_size_all), blk_size, h2)
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

mm = 0
'''
beta_init = np.array(beta_sum)
for kk in range(n_blk):
    if blk_size[kk] == 0:
        continue
    else:
        idx_blk = range(mm, mm + blk_size[kk])
        beta_init[idx_blk] = inv_ld_blk[kk].dot(np.array(beta_sum)[idx_blk])
        mm += blk_size[kk]
beta_init = beta_init * math.sqrt(h2 / np.sum(beta_init**2))*10
'''
beta_init = beta_init * math.sqrt(h2 / np.sum(beta_init**2)) * 10

n_brunin_input = 500
n_iter_input = 1000
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
result_df.to_csv('../coef_standardized.txt', sep='\t', index=False)

Lambda1 = np.array([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
Lambda2 = np.array([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3])
for lambda1 in Lambda1:
    for lambda2 in Lambda2:
        mm = 0
        coef_new = np.ones(coef.shape[0])
        for kk in range(len(blk_size)):
            if blk_size[kk] == 0:
                continue
            else:
                idx_blk = range(mm, mm + blk_size[kk])
                coef_new[idx_blk] = np.linalg.solve(np.identity(blk_size[kk]) + lambda1*ld_blk[0][kk] + lambda2*ld_blk[1][kk], coef[idx_blk] + lambda1*beta_sum[0][idx_blk] + lambda2*beta_sum[1][idx_blk])
                mm += blk_size[kk]
        result_df['BETA'] = coef_new * scale[0]
        result_df.to_csv(''.join(['../coef_EUR_lambda1_'+ str(lambda1) + '_lambda2_', str(lambda2),'.txt']), sep='\t', index=False)
        result_df['BETA'] = coef_new * scale[1]
        result_df.to_csv(''.join(['../coef_AFR_lambda1_' + str(lambda1) + '_lambda2_', str(lambda2), '.txt']), sep='\t',
                         index=False)

Res = np.ones(coef_samples.shape[1])
for i in range(coef_samples.shape[1]):
    res = 0
    coef_tmp = coef_samples[:,i]
    mm = 0
    for kk in range(len(blk_size)):
        if blk_size[kk] == 0:
            continue
        else:
            idx_blk = range(mm, mm + blk_size[kk])
            res = res - 2 * np.sum(beta_sum[1][idx_blk] * coef_tmp[idx_blk]) + \
                  np.sum(beta_sum[1][idx_blk] * ld_blk[1][kk].dot(beta_sum[1][idx_blk]))
            mm += blk_size[kk]
    Res[i] = res

Percentile = np.array([20, 30, 40, 50, 60, 70, 80, 90])
for percentile in Percentile:
    threshold = np.percentile(Res, percentile)
    coef_out = np.mean(coef_samples[:, Res < threshold], axis=1)


def constraint(beta, *args):
    blk_size, beta_sum, ld_blk, c = args
    res = 0; mm = 0
    for kk in range(len(blk_size)):
        if blk_size[kk] == 0:
            continue
        else:
            idx_blk = range(mm, mm + blk_size[kk])
            res = res - 2 * np.sum(beta_sum[1][idx_blk] * beta[idx_blk]) + \
                  np.sum(beta_sum[1][idx_blk] * ld_blk[1][kk].dot(beta[idx_blk]))
            mm += blk_size[kk]
    return c - res

def obj(beta, *args):
    beta_tmp = args
    return np.sum((beta_tmp - beta)**2)

Percentile = np.array([60, 65, 70, 75, 80, 85, 90])
for percentile in Percentile:
    coef_proj_result = coef_samples.copy()
    for i in range(coef_samples.shape[1]):
        bound = np.max(Res) * percentile / 100
        coef_tmp = coef_samples[:, i]
        cons = ({'type': 'ineq', 'fun': constraint, 'args': (blk_size, beta_sum, ld_blk, bound)})
        coef_proj = minimize(obj, coef_tmp, args=(coef_tmp), method='SLSQP', constraints=cons)
        coef_proj_result[:, i] = coef_proj['x']
    coef_out = np.mean(coef_proj_result, axis=1)
    result_df['BETA'] = coef_out * scale[0]
    result_df.to_csv(''.join([output, '/coef_EUR_percentile_' + str(percentile) + '.txt']),
                     sep='\t', index=False)
    result_df['BETA'] = coef_out * scale[1]
    result_df.to_csv(''.join([output, '/coef_AFR_percentile_' + str(percentile) + '.txt']),
                     sep='\t', index=False)



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

