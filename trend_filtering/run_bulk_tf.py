import sys
sys.path.append('../')
import string
import random
import time
from simulations.generate_sims import generate_conditional_piecewise_paths, apply_function_to_paths
from trend_filtering.test_adaptive_tf import test_adaptive_tf

def run_bulk_trend_filtering(prior,sim_style,m=500,n=25,verbose=True):
    """Solves Bulk Trend Filtering problems
    
        m: Number of simulations
        n: Length of each path
    
    """
    start_time=time.time()
    prior=prior[:m]
    
    # generate samples
    samples=generate_conditional_piecewise_paths(prior,sim_style,n_sims=n)

    random_letters = ''.join(random.choice(string.ascii_uppercase) for i in range(5))
    exp_name=f"L1_Trend_Filter_{random_letters}"

    if verbose:
        print("Running {m} simulations of length {n}".format(n=m,m=n))
        print("Experiment is {sim_style} ".format(sim_style=sim_style))

    
    
    # apply tf to each path with specified flags
    flags={'include_cv':True,'plot':True,'verbose':True,'bulk':True,'log_mlflow':True}
    results=apply_function_to_paths(samples,test_adaptive_tf,exp_name=exp_name,flags=flags)

    end_time=time.time()
    if verbose:
        print("Total time taken: {time} for {n} simulations is".format(time=end_time-start_time,n=n))

    return 
