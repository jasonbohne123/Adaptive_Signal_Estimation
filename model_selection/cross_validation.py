import sys

sys.path.append("../estimators")

from base_estimator import Base_Estimator


class Cross_Validation:
    def __init__(self, estimator: Base_Estimator, k=5, verbose=False):

        self.estimator = estimator
        self.k = k
        self.verbose = verbose
