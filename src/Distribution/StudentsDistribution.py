import numpy as np
import scipy as scp

class StudentsDistribution():
    
    def __init__(self, df=None):
        self.df = df
    
    def loglik_resid(self, returns, variance, params):
        self.df = params[-1]
        # loglikelihood from the standardized residuals 
        std_resid = returns / np.sqrt(variance)
        x = np.sqrt(self.df / (self.df - 2)) * std_resid
        return np.log(np.sqrt(self.df / (self.df - 2))) + scp.stats.t.logpdf(x, df=self.df)
    
    def ppf(self, q, params):
        self.df = params[-1]
        # quantile function
        return scp.stats.t.ppf(q, df=self.df) / np.sqrt(self.df / (self.df - 2))
    
    def random_draw(self, mu=0.0, std=1.0):
        return scp.stats.t.rvs(df=self.df, loc=mu, scale=std, size=1)[0] / np.sqrt(self.df / (self.df - 2))