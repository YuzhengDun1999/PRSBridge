import numpy as np
import scipy as sp

def generate_prs_woodbury(obs_prec, ld, sqrt_ld, eigenval, eigenvec, obs_num, prior_prec_sqrt, z, seed=None):
    if seed is not None:
        np.random.seed(seed)

    temp1 = np.random.randn(sqrt_ld.shape[1])
    temp2 = np.random.randn(ld.shape[0])
    v = np.sqrt(obs_num * obs_prec) * sqrt_ld.dot(temp1) \
        + prior_prec_sqrt * temp2
    b = z + v
    temp = (1/prior_prec_sqrt)**2 * b
    scaled_eigenvec = 1/prior_prec_sqrt[:, np.newaxis] * eigenvec
    to_be_inverted = np.diag(1/(obs_prec * obs_num * eigenval)) + scaled_eigenvec.T.dot(scaled_eigenvec)
    beta = temp - (1/prior_prec_sqrt)**2 * eigenvec.dot(sp.linalg.cho_solve(sp.linalg.cho_factor(to_be_inverted), eigenvec.T.dot(temp)))

    return beta
'''
start = time.time()
exp1 = np.linalg.solve(to_be_inverted, eigenvec.T.dot(temp))
time.time()-start

start = time.time()
c, low = sp.linalg.cho_factor(to_be_inverted)
exp2 = sp.linalg.cho_solve((c, low), eigenvec.T.dot(temp))
time.time()-start
'''