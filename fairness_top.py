import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--chr', dest='chr_input', type=int, help='chr')
parser.add_argument('--alpha', dest='alpha_input', type=float, help='alpha')
parser.add_argument('--hfile', dest='heritability', help='hertability file location')
parser.add_argument('--output', dest='output', help='output file location')
args = parser.parse_args()
#eigen_rm = args.percent_input
chr = args.chr_input
alpha = args.alpha_input
hfile = args.heritability
output = args.output

ldsc_file = hfile + '_h2.txt'
ld_blk = {}; sumdat = {}
Race = ['EUR', 'AFR']
n_pop = len(Race)
n_blk = np.loadtxt(
    '/dcs04/nilanjan/data/ydun/MultiEthnic/EUR/ref_hapmap3/chr' + str(chr) + '/block.txt').astype(int)
eigen_rm = 0.5
n_blk = np.max(n_blk)
for i in range(n_pop):
    ldblk_dir = '/dcs04/nilanjan/data/ydun/MultiEthnic/' + Race[i]  + '/ref_hapmap3/chr' + str(chr)
    sumdat_file = '/dcs04/nilanjan/data/ydun/MultiEthnic/' + Race[i]  + '/HDL/data_Rcov/chr' + str(chr) + '/sumdat-1kg-block'
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

coef = np.array(pd.read_csv(output + '/coef_standardized.txt', sep='\t')['BETA'])
coef_new = coef.copy()
Coef_Percentile = np.array([99.5, 99, 95, 90, 85, 80])
Lambda1 = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
Lambda2 = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
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
            result_df.to_csv(''.join([output, '/coef_EUR_percent_', coef_percentile, '_lambda1_' + str(lambda1) + '_lambda2_', str(lambda2), '.txt']),
                             sep='\t', index=False)
            result_df['BETA'] = coef_new * scale[1]
            result_df.to_csv(''.join([output, '/coef_AFR_percent_', coef_percentile, '_lambda1_' + str(lambda1) + '_lambda2_', str(lambda2), '.txt']),
                             sep='\t', index=False)
