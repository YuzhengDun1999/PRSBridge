from pysnptools.snpreader import Bed
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bfile', dest='bfile', help='bfile')
args = parser.parse_args()
bfile = args.bfile

geno = Bed(bfile, count_A1=False)
row_count = geno.row_count
col_count = geno.col_count
sub_geno = geno.read().val
sub_mean = np.nanmean(sub_geno, axis=0)
nanidx = np.where(np.isnan(sub_geno))
sub_geno[nanidx] = sub_mean[nanidx[1]]
sub_std = np.std(sub_geno, axis=0)
geno_std = np.nan_to_num((sub_geno - sub_mean) / sub_std)
cov = np.dot(geno_std.T, geno_std) / (geno_std.shape[0])
np.save(bfile + '.npy', cov)
