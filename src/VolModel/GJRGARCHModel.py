import numpy as np
from VolModel.BASEModel import BASEModel
from Distribution.NormalDistribution import NormalDistribution
from Distribution.StudentsDistribution import StudentsDistribution

class GJRGARCHModel(BASEModel):
    def get_variance(self, returns, params):
        def ind(X, if_pos):
            # indicator function (1 if [...], 0 otherwise)
            if X > 0:
                return if_pos
            else:
                return 1 - if_pos
        
        omega = params[0]
        alpha = params[1]
        beta = params[3]
        gamma = params[2]
    
        # initiate the variance process
        variance = [np.var(returns)]
        for t in range(1, returns.shape[0]):
    
            var_t = (omega
                     + beta * variance[t - 1]
                     + gamma * ind(returns[t - 1], 0) * returns[t - 1] ** 2
                     + alpha * returns[t - 1] ** 2)
    
            variance.append(var_t)
    
        return np.array(variance)
    
    def init_params(self):
        # outputs starting values for the optimization algorithms 
        if isinstance(self.distribution, NormalDistribution):
            return [0.0, 0.02, 0.0, 0.98]
        elif isinstance(self.distribution, StudentsDistribution):
            return [0.0, 0.02, 0.0, 0.98, 5]
    
    def init_bounds(self):
        # outputs bounds for the optimization algorithms 
        if isinstance(self.distribution, NormalDistribution):
            return [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        elif isinstance(self.distribution, StudentsDistribution):
            return [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (2.0 + 1e-6, 100)]
    
    def constraints(self, params):
        # outputs constraints for the optimization algorithms
        alpha = params[1]
        beta = params[2]
        eps = 1e-6

        return 1 - alpha - beta - eps
    
    def config_name(self):
        if isinstance(self.distribution, NormalDistribution):
            dist_name = 'Normal'
        elif isinstance(self.distribution, StudentsDistribution):
            dist_name = 'Student'
        return 'GJR-GARCH ' + dist_name