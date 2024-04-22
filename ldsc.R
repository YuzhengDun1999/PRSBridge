library(bigsnpr)
Args <- commandArgs()
beta_file = Args[6]##standardized beta file
ldsc_file = Args[7]##ldsc_file
sample_size = as.numeric(Args[8])##GWAS sample size
#gamma_out = Args[9]##initialized gamma file output
beta = read.csv(beta_file, header = FALSE)
ldsc = read.csv(ldsc_file, header = FALSE)
chi2 = beta[,1]^2 * sample_size
ld_score = ldsc[,1]
ldsc <- snp_ldsc(   ld_score, 
                    length(ld_score), 
                    chi2 = chi2,
                    sample_size = sample_size
                    )
names(ldsc) = NULL
print(ldsc[3:4])