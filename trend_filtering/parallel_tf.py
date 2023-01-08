from trend_filtering.adaptive_tf import adaptive_tf


# This is the function that we want to parallelize
def run_adaptive_tf(y, t, lambda_p, k):
    return adaptive_tf(y, t, lambda_p, k)
