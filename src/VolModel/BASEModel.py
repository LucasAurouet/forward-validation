import numpy as np
import scipy as scp
import utils

class BASEModel():
    
    def __init__(self, distribution):
        # storing variables in the object
        self.distribution = distribution
        self.returns = None

    def fit_mle(self, returns, show):
        # fits the MLE
        # scp.optimize.minimize takes the objective function as input
        # neg_loglik() is the objective function
        self.returns = returns
        def neg_loglik(params):
            # compute the negative loglikelihood for a given set of parameters
            # uses the general form loglik - 0.5 * log(variance)
            variance = self.get_variance(self.returns, params)
            loglik = self.distribution.loglik_resid(self.returns, variance, params)
            
            return - np.sum(loglik - 0.5 * np.log(variance)) 
        
        # minimization algorithm
        # initial values are given by the init_params() method
        # same for bounds and constraints 
        opt = scp.optimize.minimize(neg_loglik,
                                    x0=self.init_params(),
                                    method='SLSQP',
                                    bounds=self.init_bounds(),
                                    constraints={'type': 'ineq', 
                                                 'fun': self.constraints
                                                 },
                                    options={'disp': show,
                                             'eps': 1e-3
                                             }
                                    )
        # store the optimal parameters
        self.mle_params = opt.x
        return opt.x


    def fit_fv(self, returns, VaR_level, show):
        # Fits the Forward Validation
        self.VaR_level = VaR_level
        self.show = show
        self.returns = returns
        
        def LRcc(params):
            # objective function (conditionl coverage tests)
            VaR = self.get_valueatrisk(self.returns, params, self.VaR_level)
            violations = utils.count_violations(self.returns, VaR)
            b_loglik = utils.binomial_loglik(violations, self.VaR_level)
            m_loglik = utils.markov_loglik(violations)
            
            if self.show:
                print(params, b_loglik + m_loglik)

            return b_loglik + m_loglik

        # minimization algorithm
        # initial values are given by the init_params() method
        # same for bounds and constraints 
        opt = scp.optimize.basinhopping(LRcc,
                                        x0=self.init_params(),
                                        niter=5,
                                        T=0.01,
                                        stepsize=0.005,
                                        niter_success=5,
                                        disp=show,
                                        minimizer_kwargs={
                                            'method': 'SLSQP',
                                            'bounds': self.init_bounds(),
                                            'options': {'eps': 1e-3},
                                            'constraints': {
                                                'type': 'ineq', 
                                                'fun': self.constraints
                                                }
                                            }
                                        )
        # opt = scp.optimize.differential_evolution(LRcc,
        #                                           bounds=self.init_bounds(),
        #                                           strategy='best1bin',
        #                                           maxiter=10,
        #                                           popsize=5,
        #                                           tol=1e-6,
        #                                           mutation=(0.5, 1),
        #                                           recombination=0.7,
        #                                           init='latinhypercube',
        #                                           polish=True,
        #                                           x0=self.init_params())
        self.fv_params = opt.x
        return opt.x
    
    def get_valueatrisk(self, returns, params, VaR_level):
        # computes VaR from given parameters and given VaR level
        variance = self.get_variance(returns, params)
        f = self.distribution.ppf(VaR_level, params)
        
        return f * np.sqrt(variance)

   
