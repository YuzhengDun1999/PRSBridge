import math

import numpy as np
import scipy as sp
import time
from warnings import warn
from .random import BasicRandom
from .reg_coef_sampler import SparseRegressionCoefficientSampler
from .prior import RegressionCoefPrior
from .gibbs_util import MarkovChainManager, SamplerOptions


class PrsBridge():
    """ Implement Gibbs sampler for Bayesian bridge PRS. """

    def __init__(self, model, prior=RegressionCoefPrior()):
        """
        Parameters
        ----------
        model : PRS object
        prior : RegressionCoefPrior object
        """
        self.beta_sum = model.beta_sum
        self.ld = model.ld
        self.sqrt_ld = model.sqrt_ld
        self.eigenval_blk = model.eigenval_blk
        self.eigenvec_blk = model.eigenvec_blk
        self.obs_num = model.obs_num
        self.blk_size = model.blk_size
        self.n_pred = len(self.beta_sum) # dimension of beta
        self.n_unshrunk = prior.n_fixed
        self.prior_sd_for_unshrunk = prior.sd_for_fixed.copy()
        self.h2 = model.h2

        self.model = model
        self.prior = prior
        self.rg = BasicRandom()
        self.manager = MarkovChainManager(
            self.n_pred, self.n_unshrunk, model.name
        )
        '''
        self.manager = MarkovChainManager(
            self.n_obs, self.n_pred, self.n_unshrunk, model.name
        )
        '''

    def gibbs(self, n_iter, n_burnin=0, thin=1, seed=None,
              init={'global_scale': 0.1}, params_to_save=('coef', 'global_scale'),
              coef_sampler_type=None, n_status_update=0,
              options=None, _add_iter_mode=False, max_iter=500, update_alpha=True):
        """ Generate posterior samples under the specified model and prior.

        Parameters
        ----------
        n_iter : int
            Total number of MCMC iterations i.e. burn-ins + saved posterior draws
        n_burnin : int
            Number of burn-in samples to be discarded
        thin : int
            Number of iterations per saved samples for "thinning" MCMC to reduce
            the output size. In other words, the function saves an MCMC sample
            every `thin` iterations, returning floor(n_iter / thin) samples.
        seed : int
            Seed for random number generator.
        init : dict of numpy arrays
            Specifies, partially or completely, the initial state of Markov chain.
            The partial option allows either specifying the global scale or
            regression coefficients. The former is the recommended default option
            since the global scale parameter is easier to choose, representing
            the prior expected magnitude of regression coefficients and . (But
            the latter might make sense if some preliminary estimate of coefficients
            are available.) Other parameters are then initialized through a
            combination of heuristics and conditional optimization.
        coef_sampler_type : {None, 'cholesky', 'cg', 'hmc'}
            Specifies the sampling method used to update regression coefficients.
            If None, the method is chosen via a crude heuristic based on the
            model type, as well as size and sparsity level of design matrix.
            For linear and logistic models with large and sparse design matrix,
            the conjugate gradient sampler ('cg') is preferred over the
            Cholesky decomposition based sampler ('cholesky'). For other
            models, only Hamiltonian Monte Carlo ('hmc') can be used.
        params_to_save : {'all', tuple or list of str}
            Specifies which parameters to save during MCMC iterations. By default,
            the most relevant parameters --- regression coefficients,
            global scale, posterior log-density --- are saved. Use all to save
            all the parameters (but beaware of the extra memory requirement),
            including local scale and, depending on the model, precision (
            inverse variance) of observations.
        n_status_update : int
            Number of updates to print on stdout during the sampler run.
        max_iter : maximum iteration for conjugate gradient method.

        Other Parameters
        ----------------
        options : None, dict, SamplerOptions
            SamplerOptions class or a dict whose keywords are used as inputs
            to the class.

        Returns
        -------
        samples : dict
            Contains MCMC samples of the parameters as specified by
            **params_to_save**. The last dimension of the arrays correspond
            to MCMC iterations; for example,
            :code:`samples['coef'][:, 0]`
            is the first MCMC sample of regression coefficients.
        mcmc_info : dict
            Contains information on the MCMC run and sampler settings, enough in
            particular to reproduce and resume the sampling process.
        """

        if not isinstance(options, SamplerOptions):
            options = SamplerOptions.pick_default_and_create(
                coef_sampler_type, options
            )

        if not _add_iter_mode:
            self.rg.set_seed(seed)
            self.reg_coef_sampler = SparseRegressionCoefficientSampler(
                self.n_pred, self.prior_sd_for_unshrunk,
                options.coef_sampler_type, options.curvature_est_stabilized,
                self.prior.slab_size
            )

        if params_to_save == 'all':
            params_to_save = (
                'coef', 'local_scale', 'global_scale'
            )

        n_status_update = min(n_iter, n_status_update)
        start_time = time.time()
        self.manager.stamp_time(start_time)


        # Initial state of the Markov chain
        coef, obs_prec, lscale, gscale, init, initial_optim_info = \
            self.initialize_chain(init, self.prior.bridge_exp)

        # Pre-allocate
        if update_alpha == True:
            alpha_iter = 0; alpha_burnin = 20; alpha_max_iter = 50; coef_tmp_samples = np.ones([alpha_max_iter - alpha_burnin, coef.shape[0]])
        samples = {}
        sampling_info = {}
        self.manager.pre_allocate(
            samples, sampling_info, n_iter - n_burnin, thin, params_to_save,
            options.coef_sampler_type
        )

        # Start Gibbs sampling
        for mcmc_iter in range(1, n_iter + 1):

            #gscale = np.sqrt(self.h2 * math.factorial(1/self.prior.bridge_exp-1)/ math.factorial(3/self.prior.bridge_exp-1))

            #obs_prec = 1 / (1 - self.h2)
            obs_prec = 1
            coef, info = self.update_regress_coef(
                obs_prec, gscale, lscale, self.blk_size, coef, options.coef_sampler_type, max_iter
            )
            mm = 0
            '''
            for kk in range(len(self.blk_size)):
                if self.blk_size[kk] == 0:
                    continue
                else:
                    idx_blk = range(mm, mm + self.blk_size[kk])
                    coef[idx_blk] = self.proj_blk[kk].dot(coef[idx_blk])
                    mm += self.blk_size[kk]
            '''

            #obs_prec = self.update_obs_precision(coef)
            obs_prec = 1/(1-self.h2)
            obs_prec = 1

            # Draw from gscale | coef and then lscale | gscale, coef.
            # (The order matters.)
            #gscale = np.sqrt(self.h2 * math.gamma(1/self.prior.bridge_exp)/ math.gamma(3/self.prior.bridge_exp)/len(coef))

            gscale = self.update_global_scale(
                gscale, coef[self.n_unshrunk:], self.prior.bridge_exp,
                coef_expected_magnitude_lower_bd=0, method=options.gscale_update
            )


            lscale = self.update_local_scale(
                gscale, coef[self.n_unshrunk:], self.prior.bridge_exp)

            if update_alpha == True:
                if (mcmc_iter < n_burnin) & (mcmc_iter > 200):
                    coef_tmp_samples, alpha_iter = self.update_bridge_exp(
                        alpha_iter, coef / gscale, coef_tmp_samples, alpha_burnin, alpha_max_iter
                    )

            self.manager.store_current_state(
                samples, mcmc_iter, n_burnin, thin, coef, lscale, gscale,
                self.prior.bridge_exp, obs_prec, params_to_save
            )
            self.manager.store_sampling_info(
                sampling_info, info, mcmc_iter, n_burnin, thin,
                options.coef_sampler_type
            )
            self.manager.print_status(n_status_update, mcmc_iter, n_iter)

        runtime = time.time() - start_time

        if self.prior._gscale_paramet == 'coef_magnitude':
            gscale, lscale = \
                self.prior.adjust_scale(gscale, lscale, to='coef_magnitude')
            gscale_samples = samples.get('global_scale', 0.)
            lscale_samples = samples.get('local_scale', 0.)
            self.prior.adjust_scale(
                gscale_samples, lscale_samples, to='coef_magnitude'
            ) # Modify in place.

        _reg_coef_sampling_info = None
        mcmc_info = {
            'init': init,
            'n_iter': n_iter,
            'n_burnin': n_burnin,
            'thin': thin,
            'seed': seed,
            'n_coef_wo_shrinkage': self.n_unshrunk,
            'prior_sd_for_unshrunk': self.prior_sd_for_unshrunk,
            'bridge_exponent': self.prior.bridge_exp,
            'coef_sampler_type': options.coef_sampler_type,
            'saved_params': params_to_save,
            'runtime': runtime,
            'options': options.get_info(),
            '_init_optim_info': initial_optim_info,
            '_reg_coef_sampling_info': sampling_info,
            '_random_gen_state': self.rg.get_state(),
            '_reg_coef_sampler_state': self.reg_coef_sampler.get_internal_state()
        }

        return samples, mcmc_info

    def get_sampling_info_keys(self, sampling_method):
        if sampling_method == 'cg':
            keys = ['n_cg_iter']
        elif sampling_method in ['hmc', 'nuts']:
            keys = [
                'stepsize', 'n_hessian_matvec', 'n_grad_evals',
                'stability_limit_est', 'stability_adjustment_factor',
                'instability_detected'
            ]
            if sampling_method == 'hmc':
                keys += ['n_integrator_step', 'accepted', 'accept_prob']
            else:
                keys += ['tree_height', 'ave_accept_prob']
        else:
            keys = []
        return keys

    def store_sampling_info(
            self, sampling_info, info, mcmc_iter, n_burnin, thin, sampling_method):

        if mcmc_iter <= n_burnin or (mcmc_iter - n_burnin) % thin != 0:
            return

        index = math.floor((mcmc_iter - n_burnin) / thin) - 1
        for key in self.get_sampling_info_keys(sampling_method):
            sampling_info[key][index] = info[key]

    def initialize_chain(self, init, bridge_exp):
        """ Choose the user-specified state if provided, the default ones otherwise."""

        valid_param_name \
            = ('coef', 'local_scale', 'obs_prec', 'global_scale')
        for key in init:
            if key not in valid_param_name:
                warn("'{:s}' is not a valid parameter name and "
                     "will be ignored.".format(key))

        coef_only_specified = 'coef' in init and ('global_scale' not in init)

        if 'coef' in init:
            coef = init['coef']
            if not len(coef) == self.n_pred:
                raise ValueError('Invalid initial length of regression coefficient.')
        else:
            coef = np.zeros(self.n_pred)
            if self.model.name in ('linear', 'logit'):
                coef[0] = self.model.calc_intercept_mle()

        obs_prec = self.initialize_obs_precision(init, coef)

        if coef_only_specified:
            gscale = self.update_global_scale(
                None, coef[self.n_unshrunk:], bridge_exp,
                method='optimize'
            )
            lscale = self.update_local_scale(
                gscale, coef[self.n_unshrunk:], bridge_exp
            )
        else:
            if 'global_scale' not in init:
                raise ValueError("Initial global scale must be specified when "
                                 "coefficients aren't specified.")
            if self.prior._gscale_paramet == 'raw':
                warn("Using the raw global scale parametrization; make sure that "
                     "the specified initial value is scaled accordingly.")
            gscale = init['global_scale']
            if 'local_scale' in init:
                lscale = init['local_scale']
                if not len(lscale) == (self.n_pred - self.n_unshrunk):
                    raise ValueError('Invalid initial length of local scale parameter')
            else:
                lscale = np.ones(self.n_pred - self.n_unshrunk) / gscale

        if self.prior._gscale_paramet == 'coef_magnitude':
            # Gibbs sampler requires the raw parametrization, though
            # technically only gscale * lscale matters due to the update order.
            gscale, lscale \
                = self.prior.adjust_scale(gscale, lscale, to='raw')
        '''
        if 'coef' not in init:
            # Optimize coefficients and then update other parameters
            coef, info = self.reg_coef_sampler.search_mode(
                coef, lscale, gscale, self.model
            )
            lscale = self.update_local_scale(
                gscale, coef[self.n_unshrunk:], bridge_exp
            )
            optim_info = {
                key: info[key] for key in ['is_success', 'n_design_matvec', 'n_iter']
            }
        else:
            optim_info = None
        '''
        init = {
            'coef': coef,
            'local_scale': lscale,
            'global_scale': gscale
        }
        optim_info = None

        return coef, obs_prec, lscale, gscale, init, optim_info

    def initialize_obs_precision(self, init, coef):
        if 'obs_prec' in init:
            obs_prec = np.ascontiguousarray(init['obs_prec'])
            # Cython requires a C-contiguous array.
            if not len(obs_prec) == self.n_obs:
                raise ValueError('An invalid initial state.')
        elif self.model.name == 'linear':
            obs_prec = np.mean(
                (self.model.y - self.model.design.dot(coef)) ** 2) ** -1
        else:
            obs_prec = None

        obs_prec = 1
        return obs_prec

    def update_regress_coef(self, obs_prec, gscale, lscale, blk_size, coef, sampling_method, max_iter):

        if sampling_method in ('cholesky', 'cg', 'woodbury', 'univariate'):
            coef, info = self.reg_coef_sampler.sample_gaussian_posterior_prs(
                self.beta_sum, self.ld, self.sqrt_ld, self.eigenval_blk, self.eigenvec_blk, self.obs_num,
                obs_prec, gscale, lscale, blk_size, coef, sampling_method, max_iter
            )

        else:
            raise NotImplementedError()

        return coef, info

    def update_obs_precision(self, coef):

        obs_prec = None
        mm = 0
        resid = np.ones(coef.shape[0])
        scale = 0
        for kk in range(len(self.blk_size)):
            if self.blk_size[kk] == 0:
                continue
            else:
                idx_blk = range(mm, mm + self.blk_size[kk])
                resid[idx_blk] = self.beta_sum[idx_blk] - self.ld[kk].dot(coef[idx_blk])
                scale = scale + np.diag(self.obs_num[idx_blk])
                mm += blk_size[kk]

        resid = self.beta_sum - self.ld.dot(coef)
        scale = self.obs_num * resid.T.dot(self.inv_ld).dot(resid) / 2
        obs_var = scale / self.rg.np_random.gamma(len(coef) / 2, 1)
        obs_prec = 1 / obs_var

        return obs_prec


    def update_global_scale(
            self, gscale, beta_with_shrinkage, bridge_exp,
            coef_expected_magnitude_lower_bd=0, method='sample'):
        # :param method: {"sample", "optimize", None}

        if beta_with_shrinkage.size == 0:
            return 1. # arbitrary float value as a placeholder

        lower_bd = coef_expected_magnitude_lower_bd \
                   / self.prior.compute_power_exp_ave_magnitude(bridge_exp)
            # Solve for the value of global shrinkage such that
            # (expected value of regress_coef given gscale) = coef_expected_magnitude_lower_bd.

        if method == 'optimize':
            gscale = self.monte_carlo_em_global_scale(
                beta_with_shrinkage, bridge_exp)

        elif method == 'sample':
            # Conjugate update for phi = 1 / gscale ** bridge_exp
            if np.count_nonzero(beta_with_shrinkage) == 0:
                gscale = 0
            else:
                prior_param = self.prior.param['gscale_neg_power']
                shape, rate = prior_param['shape'], prior_param['rate']
                shape += beta_with_shrinkage.size / bridge_exp
                rate += np.sum(np.abs(beta_with_shrinkage) ** bridge_exp)
                phi = self.rg.np_random.gamma(shape, scale=1 / rate)
                gscale = 1 / phi ** (1 / bridge_exp)

        if (method is not None) and gscale < lower_bd:
            gscale = lower_bd
            warn(
                "The global shrinkage parameter update returned an unreasonably "
                "small value. Returning a specified lower bound value instead."
            )

        return gscale

    def monte_carlo_em_global_scale(
            self, beta_with_shrinkage, bridge_exp):
        """ Maximize the likelihood (not posterior conditional) 'coef | gscale'. """
        phi = len(beta_with_shrinkage) / bridge_exp \
              / np.sum(np.abs(beta_with_shrinkage) ** bridge_exp)
        gscale = phi ** - (1 / bridge_exp)
        return gscale

    def update_local_scale(self, gscale, beta_with_shrinkage, bridge_exp):
        if bridge_exp == 2:
            return np.ones(beta_with_shrinkage.size) / np.sqrt(2)

        lscale_sq = .5 / self.rg.tilted_stable(
            bridge_exp / 2, (beta_with_shrinkage / gscale) ** 2
        )
        lscale = np.sqrt(lscale_sq)

        # TODO: Pick the lower and upper bound more carefully.
        if np.any(lscale == 0):
            warn(
                "Local scale parameter under-flowed. Replacing with a small number.")
            lscale[lscale == 0] = 10e-16
        elif np.any(np.isinf(lscale)):
            warn(
                "Local scale parameter over-flowed. Replacing with a large number.")
            lscale[np.isinf(lscale)] = 2.0 / gscale

        return lscale

    def update_bridge_exp(self, alpha_iter, coef_tmp, coef_tmp_samples, alpha_burnin=20, alpha_max_iter=50):
        alpha_iter = alpha_iter + 1
        if alpha_iter < alpha_burnin:
            return coef_tmp_samples, alpha_iter
        elif alpha_iter < alpha_max_iter:
            coef_tmp_samples[alpha_iter - alpha_burnin, :] = np.abs(coef_tmp)
        elif alpha_iter == alpha_max_iter:
            alpha_iter = 0
            self.prior.bridge_exp = self.stochastic_gradient_descent(coef_tmp_samples, self.prior.bridge_exp)
            coef_tmp_samples = np.ones([alpha_max_iter - alpha_burnin, coef_tmp.shape[0]])
        return coef_tmp_samples, alpha_iter

    def stochastic_gradient_descent(self, constant, starting_point=0.5, learning_rate=0.001, num_iterations=1000,
                                    tol=0.001):
        # constant: abs(\beta)/\tau
        alpha_values = np.arange(0.1, 0.5, 0.01)
        gradients = []; alphas = []
        for alpha in alpha_values:
            gradient = (constant.shape[1] + constant.shape[1] * sp.special.digamma(1/alpha) * 1/alpha - np.sum(np.multiply(constant ** alpha, np.log(constant))) * alpha / constant.shape[0])
            alphas.append(alpha)
            gradients.append(gradient)
        index = np.argmin(np.abs(gradients))
        best_alpha = alphas[index]

        return best_alpha