### Estimators

Within this repo, there are a number of estimators designed for filtering and smoothing of continous data. All estimators can be found within the estimators directory. The following estimators are currently implemented:
- Regression Splines
- Kernel Regression
- Kernel Regression with Robust Extensions (Median of Means)
- Segmented Regression 

All estimators inherit from the base class ```Base_Estimator```. This class provides the `fit` and `estimate` method. Estimators I am hoping to soon implement include:

- Variable Knot Splines
- Smoothing Splines


Estimators that do not require hyperparameters (regression splines, segmented regression) are trained on initialization. Estimators that require hyperparameters (kernel regression, trend filtering) are trained using the `fit` method. The `estimate` method is used to estimate the signal at a given time point.