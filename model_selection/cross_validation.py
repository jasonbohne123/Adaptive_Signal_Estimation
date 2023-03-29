import sys

sys.path.append("../../estimators")
sys.path.append("../helpers")

from collections import defaultdict

from base_estimator import Base_Estimator
from helpers.cross_validation_constants import get_cv_constants


class K_Fold_Cross_Validation:
    def __init__(self, estimator: Base_Estimator):

        self.estimator = estimator
        self.k = k
        self.verbose = verbose

        self.k_folds = get_cv_constants()["k_folds"]
        self.verbose = get_cv_constants()["verbose"]
        self.grid_spacing = get_cv_constants()["grid_spacing"]
        self.grid_size = get_cv_constants()["grid_size"]

        self.n = len(self.estimator.x)
        self.cv_size = int(self.n / self.k_folds)

        self.x = self.estimator.x
        self.y = self.estimator.y

    def generate_grid(self):
        if self.grid_spacing == "log":
            grid = np.geomspace(get_cv_constants()["cv_grid_lb"], 1, self.grid_size)
        elif self.grid_spacing == "linear":
            grid = np.linspace(get_cv_constants()["cv_grid_lb"], 1, self.grid_size)
        else:
            raise ValueError("Grid spacing must be either log or linear")

        return grid

    def cross_validation(self):

        results = defaultdict(float)

        # iterate over multiple cross validation indices to prevent overfitting  to oos data
        for i in range(self.k_folds):
            if self.verbose:
                print(f"Performing  {i} out of {cv_iterations} iterations of cross validation")

            # get in-sample and out-of-sample indices per each iteration
            # (randomly select cv_size indices which is ideal otherwise is just extrapolation)
            is_index = np.sort(np.random.choice(len(self.x), size=self.cv_size, replace=False))
            oos_index = np.sort(np.setdiff1d(np.arange(len(self.x)), is_index))

            # get in-sample and out-of-sample data
            x_is = self.x[is_index]
            x_oos = self.x[oos_index]

            # create unequally spaced time index either from original time index or from index
            is_t = D.t[is_index] if D.time_enabled else is_index

            # create respective prior for in-sample data
            is_prior = D.prior[is_index] if D.prior_enabled else np.ones(len(is_index))

            # reformulate prior here

            is_D = Difference_Matrix(len(is_index), D.k, is_t, is_prior)

            # compute lambda_max for each subproblem given difference matrix and prior
            prior_max, is_D_D = compute_lambda_max(is_D, x_is)

            for lambda_i in grid:

                # relative lambda scaled to max
                relative_scaler = lambda_i * prior_max

                if verbose:
                    print(f"Performing cross validation for lambda = {relative_scaler}")

                # solve tf subproblem

                result = adaptive_tf(x_is.reshape(-1, 1), is_D_D, lambda_p=relative_scaler, cv=True)
                status = result["status"]
                sol = result["sol"]

                if sol is None:
                    if verbose:
                        print("No solution found for lambda = {}".format(relative_scaler))
                        print("Status: {}".format(status))

                    # ignore cases where no solution is found
                    results[lambda_i] += np.inf
                    continue

                # to compute oos error we need to make the return type callable

                predictions = sol.predict(oos_index)

                # compute mse on oos test set
                oos_error = compute_error(predictions.reshape(-1, 1), x_oos, type="mse")

                # add to average oos error for each lambda
                results[lambda_i] += oos_error

        # get best lambda from all iterations
        best_prior_dict = {k: v / cv_iterations for k, v in results.items()}
        best_prior = min(best_prior_dict, key=best_prior_dict.get)

        # get original lambda_max
        is_t = D.t if D.time_enabled else None
        orig_scaler_max, D_bar = compute_lambda_max(D, x)

        return
