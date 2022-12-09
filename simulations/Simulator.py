import numpy as np
from scipy.stats import norm, uniform, lognorm, expon, gamma, beta

class Simulator():
    """ Base class for simulators.
    """
    def __init__(self,dist,rng=None,variance_scaling=10e-4):
        
        self.variance_scaling = variance_scaling
        
        if dist == 'normal':
            self.dist = norm
        elif dist == 'uniform':
            self.dist = uniform
        elif dist == 'lognormal':
            self.dist = lognorm
        elif dist == 'exponential':
            self.dist = expon
        elif dist == 'gamma':
            self.dist = gamma
        elif dist == 'beta':
            self.dist = beta
        else:
            raise ValueError('Distribution not supported')
        
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng
    
    def simulate(self,n,rng=None):
        """Simulate data from a specific distribution."""

        if rng is None:
            rng = np.random.RandomState()
        
        return self.dist.rvs(size=n,scale=self.variance_scaling,random_state=rng)


        