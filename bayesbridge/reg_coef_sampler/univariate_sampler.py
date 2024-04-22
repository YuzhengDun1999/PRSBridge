import numpy as np
import scipy as sp

def generate_prs_univariate(beta_sum, obs_prec, ld, coef, obs_num, prior_prec_sqrt, rand_gen=None):
    if rand_gen is None:
        gaussian_vec = np.random.randn(len(beta_sum))
    else:
        gaussian_vec = rand_gen.np_random.randn(len(beta_sum))
    residual = beta_sum - ld.dot(coef) + np.diag(ld) * coef
    mu = 1/(1+prior_prec_sqrt**2/obs_num) * residual
    sigma2 = 1/(obs_num+prior_prec_sqrt**2)
    beta = mu + gaussian_vec * np.sqrt(sigma2)

    return beta