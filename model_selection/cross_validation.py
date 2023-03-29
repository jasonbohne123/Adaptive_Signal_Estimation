import sys

sys.path.append("../../estimators")
sys.path.append("../helpers")

from collections import defaultdict

import numpy as np
from base_estimator import Base_Estimator
from helpers.cross_validation_constants import get_cv_constants

from evaluation_metrics.loss_functions import compute_error


class K_Fold_Cross_Validation:
    """General Class for K-Fold Cross Validation on a Base Estimator"""

    def __init__(self, estimator: Base_Estimator):

        self.estimator = estimator
        self.hyperparams = list(estimator.hypers.keys())
        self.hypermax = estimator.hyper_max

        # get cross validation constants
        self.k_folds = get_cv_constants()["k_folds"]
        self.verbose = get_cv_constants()["verbose"]
        self.grid_spacing = get_cv_constants()["grid_spacing"]
        self.grid_size = get_cv_constants()["grid_size"]

        # generate grid for cross validation
        self.grid = [self.generate_grid() * self.hypermax[hyper] for hyper in self.hyperparams][0]

        # get size of cross validation
        self.n = len(self.estimator.x)
        self.cv_size = int(self.n / self.k_folds)

        # save x and y
        self.x = self.estimator.x
        self.y = self.estimator.y

    def generate_grid(self):
        """Generates relative grid for Cross Validation -> [0,1]"""
        if self.grid_spacing == "log":
            grid = np.geomspace(get_cv_constants()["cv_grid_lb"], 1, self.grid_size)
        elif self.grid_spacing == "linear":
            grid = np.linspace(get_cv_constants()["cv_grid_lb"], 1, self.grid_size)
        else:
            raise ValueError("Grid spacing must be either log or linear")

        return grid

    def cross_validation(self):
        """Cross Validation across Grid"""

        results = defaultdict(float)

        # iterate over multiple cross validation indices to prevent overfitting  to oos data
        for i in range(self.k_folds):
            if self.verbose:
                print(f"Performing  {i} out of {self.k_folds} iterations of cross validation")

            # get in-sample and out-of-sample indices per each iteration
            # (randomly select cv_size indices which is ideal otherwise is just extrapolation)
            is_index = np.sort(np.random.choice(len(self.x), size=self.cv_size, replace=False))
            oos_index = np.sort(np.setdiff1d(np.arange(len(self.x)), is_index))

            # get in-sample and out-of-sample data
            self.x[is_index]
            x_oos = self.x[oos_index]

            for lambda_i in self.grid:

                if self.verbose:
                    print(f"Performing cross validation for lambda = {lambda_i}")

                # solve tf subproblem
                self.estimator.update_params({self.hyperparams[0]: lambda_i})
                y_hat = self.estimator.fit(warm_start=True)

                if y_hat is None:
                    if self.verbose:
                        print("No solution found for lambda = {}".format(lambda_i))

                    # ignore cases where no solution is found
                    results[lambda_i] += np.inf
                    continue

                # to compute oos error we need to make the return type callable

                estimates = self.estimator.estimate(x_oos)

                # compute mse on oos test set
                oos_error = compute_error(estimates, x_oos, type="mse")

                # add to average oos error for each lambda
                results[lambda_i] += oos_error

        # get best lambda from all iterations
        best_prior_dict = {k: v / self.k_folds for k, v in results.items()}
        best_prior = min(best_prior_dict, key=best_prior_dict.get)

        return {"lambda_": best_prior}
