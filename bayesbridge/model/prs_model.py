from .abstract_model import AbstractModel
import math
import numpy as np


class PRSModel(AbstractModel):

    def __init__(self, beta_sum, ld, sqrt_ld, eigenval_blk, eigenvec_blk, obs_num, blk_size, h2):
        self.beta_sum = beta_sum
        self.ld = ld
        self.name = 'PRS'
        self.obs_num = obs_num
        self.sqrt_ld = sqrt_ld
        self.eigenval_blk = eigenval_blk
        self.eigenvec_blk = eigenvec_blk
        self.blk_size = blk_size
        self.h2 = h2

    def compute_loglik_and_gradient(self, beta, obs_prec, loglik_only=False):

        X_beta = self.design.dot(beta)
        loglik = (
            len(self.y) * math.log(obs_prec) / 2
            - obs_prec * np.sum((self.y - X_beta) ** 2) / 2
        )
        if loglik_only:
            grad = None
        else:
            grad = obs_prec * self.design.Tdot(self.y - X_beta)
        return loglik, grad