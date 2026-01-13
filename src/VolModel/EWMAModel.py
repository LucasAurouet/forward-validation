import numpy as np
from VolModel.BASEModel import BASEModel
from Distribution.NormalDistribution import NormalDistribution
from Distribution.StudentsDistribution import StudentsDistribution

class EWMAModel(BASEModel):

    def get_variance(self, returns, params):
        beta = params[0]
        alpha = 1 - beta
        # initiate the variance process
        variance = [np.var(returns)]
        for t in range(1, returns.shape[0]):
            var_t = (beta * variance[t - 1]
                     + alpha * returns[t - 1] ** 2)
            variance.append(var_t)

        return np.array(variance)
    
    def init_params(self):
        dist = self.distribution
        # starting values for the optimization algorithms 
        if isinstance(dist, NormalDistribution):
            return [0.98]
        elif isinstance(dist, StudentsDistribution):
            return [0.98, 5]
    
    def init_bounds(self):
        dist = self.distribution
        # outputs bounds for the optimization algorithms 
        if isinstance(dist, NormalDistribution):
            return [(0.0, 1.0)]
        elif isinstance(dist, StudentsDistribution):
            return [(0.0, 1.0), (2.0 + 1e-6, 100)]
        else:
           raise ValueError('unknown distribution')
    
    def constraints(self, params):
        # outputs constraints for the optimization algorithms
        alpha = params[0]
        eps = 1e-6

        return 1 - alpha - eps
    
    def config_name(self):
        if isinstance(self.distribution, NormalDistribution):
            dist_name = 'Normal'
        elif isinstance(self.distribution, StudentsDistribution):
            dist_name = 'Student'
        return 'EWMA ' + dist_name

    

        
        
            
            
        