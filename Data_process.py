import numpy as np
import pandas as pd

def parse_bim(bim_file):
    bim_dict = {'SNP':[], 'A1':[], 'A2':[]}
    with open(bim_file + '.bim') as ff:
        for line in ff:
            ll = (line.strip()).split()
            bim_dict['SNP'].append(ll[1])
            bim_dict['A1'].append(ll[4])
            bim_dict['A2'].append(ll[5])
    return bim_dict

def parse_sumstats(ref_loc, sumstat_file):
    n_blk = int(np.max(np.loadtxt(ref_loc + '/block.txt')))
    ATGC = ['A', 'T', 'G', 'C']
    #sumstat_dict = {'chr':[], 'SNP':[], 'A1':[], 'A2':[], 'frq':[], 'P':[], 'beta':[], 'SE':[], 'N':[]}
    sumstat_dict = {'SNP': [], 'A1': [], 'A2': [], 'beta': [], 'SE': [], 'N': []}
    with open(sumstat_file) as ff:
        header = next(ff)
        for line in ff:
            ll = (line.strip()).split()
            if ll[1] in ATGC and ll[2] in ATGC:
                #if float(ll[5]) < 0.01 or float(ll[5]) > 0.99:
                #    continue
                #sumstat_dict['chr'].append(int(ll[0]))
                sumstat_dict['SNP'].append(ll[0])
                sumstat_dict['A1'].append(ll[1])
                sumstat_dict['A2'].append(ll[2])
                #sumstat_dict['frq'].append(float(ll[5]))
                #sumstat_dict['P'].append(float(ll[6]))
                sumstat_dict['beta'].append(float(ll[3]))
                sumstat_dict['SE'].append(float(ll[4]))
                sumstat_dict['N'].append(int(ll[5]))

    sumstat_df = pd.DataFrame.from_dict(sumstat_dict)
    sumstat_df = sumstat_df[sumstat_df['N'] > 0.9 * np.max(sumstat_df['N'])]
    sumstat_df = sumstat_df[(sumstat_df['beta']/sumstat_df['SE']) ** 2 / sumstat_df['N'] < 80]
    sumstat = []; SNP_list = []; A1_list = []; A2_list = []
    for blk in range(1, n_blk+1):
        ref_bim = parse_bim(ref_loc + '/block' + str(blk))
        sumstat_tmp = sumstat_df[sumstat_df['SNP'].isin(ref_bim['SNP'])].copy()
        Index = []; new_beta = []; scale = []; new_frq = []
        for i in sumstat_tmp.index:
            ref_index = ref_bim['SNP'].index(sumstat_tmp.loc[i, 'SNP'])
            Index.append(ref_index)
            scale_tmp = np.sqrt(sumstat_tmp.loc[i, 'SE'] ** 2 * sumstat_tmp.loc[i, 'N'] + sumstat_tmp.loc[i, 'beta'] ** 2)
            scale.append(scale_tmp)
            if sumstat_tmp.loc[i, 'A1'] == ref_bim['A2'][ref_index] and sumstat_tmp.loc[i, 'A2'] == ref_bim['A1'][ref_index]:
                new_beta.append(-sumstat_tmp.loc[i, 'beta']/scale_tmp)
                sumstat_tmp.loc[i, 'A2'] = ref_bim['A2'][ref_index]
                sumstat_tmp.loc[i, 'A1'] = ref_bim['A1'][ref_index]
                #new_frq.append(1 - sumstat_tmp.loc[i, 'frq'])
            elif sumstat_tmp.loc[i, 'A1'] == ref_bim['A1'][ref_index] and sumstat_tmp.loc[i, 'A2'] == ref_bim['A2'][ref_index]:
                new_beta.append(sumstat_tmp.loc[i, 'beta']/scale_tmp)
                #new_frq.append(sumstat_tmp.loc[i, 'frq'])
            else:
                new_beta.append(np.nan)
                #new_frq.append(np.nan)

        sumstat_tmp['new_BETA'] = new_beta; sumstat_tmp['scale'] = scale; sumstat_tmp['index'] = Index; #sumstat_tmp['frq'] = new_frq
        sumstat_tmp = sumstat_tmp.dropna()
        sumstat_tmp = sumstat_tmp.sort_values(['index'])
        SNP_list.extend(sumstat_tmp['SNP'].tolist())
        A1_list.extend(sumstat_tmp['A1'].tolist())
        A2_list.extend(sumstat_tmp['A2'].tolist())
        sumstat.append(sumstat_tmp)
    return sumstat, SNP_list, A1_list, A2_list

