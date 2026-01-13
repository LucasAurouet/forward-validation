import numpy as np
from VolModel.BASEModel import BASEModel
from Distribution.NormalDistribution import NormalDistribution
from Distribution.StudentsDistribution import StudentsDistribution

class GARCHModel(BASEModel):

    def get_variance(self, returns, params):
        self.true_variance = np.power(returns, 2)

        omega = params[0]
        alpha = params[1]
        beta = params[2]     

        # initiate the variance process
        variance = [np.var(returns)]
        for t in range(1, returns.shape[0]):
            var_t = (omega
                     + beta * variance[t - 1]
                     + alpha * returns[t - 1] ** 2)
            variance.append(var_t)
        return np.array(variance)
    
    def init_params(self):
        dist = self.distribution
        # outputs starting values for the optimization algorithms 
        if isinstance(dist, NormalDistribution):
            return [0.0, 0.02, 0.98]
        elif isinstance(dist, StudentsDistribution):
            return [0.0, 0.02, 0.98, 5]
        else:
           raise ValueError('unknown distribution')
    
    def init_bounds(self):
        dist = self.distribution
        # outputs bounds for the optimization algorithms 
        if isinstance(dist, NormalDistribution):
            return [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        elif isinstance(dist, StudentsDistribution):
            return [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (2.0 + 1e-6, 100)]
    
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
        return 'GARCH ' + dist_name