import sys

sys.path.append("../../estimators")


from collections import defaultdict

import numpy as np

from estimators.base_estimator import Base_Estimator
from evaluation_metrics.loss_functions import compute_error
from model_selection.helpers.cross_validation_constants import get_cv_constants
from model_selection.helpers.fit_submodel import fit_submodel


class K_Fold_Cross_Validation:
    """General Class for K-Fold Cross Validation on a Base Estimator"""

    def __init__(self, estimator: Base_Estimator):

        self.estimator = estimator
        self.estimator_name = estimator.name

        # get hyperparameters
        self.hyperparams = list(estimator.hypers.keys())
        self.hypermax = estimator.hyper_max

        self.estimator_configs = estimator.configs

        # get cross validation constants
        self.k_folds = get_cv_constants()["k_folds"]
        assert self.k_folds > 1

        self.verbose = get_cv_constants()["verbose"]
        self.grid_spacing = get_cv_constants()["grid_spacing"]
        self.grid_size = get_cv_constants()["grid_size"]

        # generate grid for cross validation
        self.relative_grid = self.generate_grid()
        # get size of cross validation
        self.n = len(estimator.x)

        # size of leave-in set
        self.cv_size = self.n - int(self.n / self.k_folds)

        # save x and y
        self.x = estimator.x
        self.y = estimator.y

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
            x_is = self.y[is_index]
            x_oos = self.y[oos_index]

            # fit submodel on in-sample data
            submodel = fit_submodel(is_index, x_is, self.estimator_configs, self.estimator_name)

            # get local lambda max
            local_lambda_max = submodel.lambda_max

            for lambda_i in self.relative_grid:

                lambda_scaled = lambda_i * local_lambda_max

                if self.verbose:
                    print(f"Performing cross validation for lambda = {lambda_scaled}")

                # iteratively update regularization param
                hyperparams = {hyper: lambda_scaled for hyper in self.hyperparams}
                submodel.update_params(hyperparams)

                if submodel.y_hat is None:

                    # ignore cases where no solution is found
                    results[lambda_i] += np.inf
                    continue

                # to compute oos error we need to make the return type callable

                estimates = submodel.estimate(oos_index)

                # compute mse on oos test set
                oos_error = compute_error(estimates, x_oos, type="mse")

                # add to average oos error for each lambda
                results[lambda_i] += oos_error

        # get best lambda from all iterations
        best_prior_dict = {k: v / self.k_folds for k, v in results.items()}

        best_prior = min(best_prior_dict, key=best_prior_dict.get)

        return {"lambda_": best_prior * self.estimator.lambda_max}
