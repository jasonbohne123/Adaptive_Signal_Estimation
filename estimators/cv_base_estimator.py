from estimators.base_estimator import Base_Estimator
from model_selection.cross_validation import K_Fold_Cross_Validation


def cv_base_estimator(estimator: Base_Estimator):
    """Cross Validation for Base Estimator"""

    # create cross validation object
    cv_estimator = K_Fold_Cross_Validation(estimator)

    # perform cross validation on grid
    optimal_params = cv_estimator.cross_validation()

    # update estimator with optimal parameters
    estimator.update_params(optimal_params)

    assert estimator.y_hat is not None

    return optimal_params
