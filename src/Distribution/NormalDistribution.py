import numpy as np
import scipy as scp
import random

class NormalDistribution():
    
    def loglik_resid(self, returns, variance, params):
        # loglikelihood from the standardized residuals 
        std_resid = returns / np.sqrt(variance)
        
        return scp.stats.norm.logpdf(std_resid)
    
    def ppf(self, q, params):
        # quantile function
        return scp.stats.norm.ppf(q)
    
    def random_draw(self, mu=0, std=1):
        return scp.stats.norm.rvs(loc=mu, scale=std, size=1)[0]