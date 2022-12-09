import numpy as np
from simulations.label_changepoints import label_changepoints
from .Simulator import Simulator

class ConditionalSimulator(Simulator):
    def __init__(self,prior,sim_style,label_style,k_points,underlying_dist=None,n_sims=1000,variance_scaling=10e-4,rng=None):
        self.prior = prior
        self.sim_style = sim_style
        self.label_style = label_style
        self.k_points = k_points
        
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.underlying_simulator = Simulator(underlying_dist,self.rng,variance_scaling)
        self.n_sims = n_sims

        self.cp_index = label_changepoints(prior, label_style,k_points)



    def simulate(self):
        """Sample from the prior distribution at cp_index"""
       
        true_processes = np.zeros((self.n_sims,len(self.prior)))

        true_processes[:,0]=np.zeros(self.n_sims)

        true_processes[:,self.cp_index] = self.underlying_simulator.simulate(len(self.cp_index),self.rng)

        return true_processes
    
 

    def evaluate_within_sample(self,cp_index,true_processes):
        """Generate known points (from style) within sampled index"""
        
        if self.sim_style == 'piecewise_constant':
            for i in range(len(cp_index)-1):
                # evaluate remaining of the interval to be constant
                true_processes[:,cp_index[i]:cp_index[i+1]] = np.multiply(true_processes[:,cp_index[i]],np.ones((self.n_sims,cp_index[i+1]-cp_index[i])))
              
        
        elif self.sim_style == 'piecewise_linear':
            for i in range(len(cp_index)-1):
                
                steps=cp_index[i+1]-cp_index[i]
                diff=2*true_processes[:,cp_index[i]]-true_processes[:,cp_index[i]-1]

                end_point=true_processes[:,cp_index[i]]+steps*diff
                true_processes[:,cp_index[i]:cp_index[i+1]] = np.linspace(true_processes[:,cp_index[i]],end_point,cp_index[i+1]-cp_index[i]).T
        else:
            raise ValueError('Invalid simulation style')
        
        
        return true_processes




# Different Simulation Styles
# --------------------------
# Piecweise Constant
# -----------------
# Piecewise Linear
# ----------------
