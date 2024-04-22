import numpy as np
import scipy as sp

def generate_prs_gaussian(beta_sum, obs_prec, ld, sqrt_ld, obs_num, prior_prec_sqrt, rand_gen=None):
    """
    Generate a multi-variate Gaussian with the mean mu and covariance Sigma of the form
        mu = Sigma z,
        Sigma^{-1} = X' diag(obs_prec) X + diag(prior_prec_sqrt) ** 2,

    Parameters
    ----------
        beta_sum : 1-d numpy array
        prior_prec_sqrt : 1-d numpy array
        obs_num : 1-d numpy array, number of individuals
    """
    Phi = obs_prec * obs_num * ld + np.diag(prior_prec_sqrt ** 2)
    Phi_chol = sp.linalg.cholesky(Phi)
    z = obs_num * beta_sum * obs_prec
    mu = sp.linalg.cho_solve((Phi_chol, False), z)
    if rand_gen is None:
        gaussian_vec = np.random.randn(len(beta_sum))
    else:
        gaussian_vec = rand_gen.np_random.randn(len(beta_sum))
    beta = mu + sp.linalg.solve_triangular(
        Phi_chol, gaussian_vec, lower=False
    )

    return beta

