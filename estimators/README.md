### Estimators

Within this repo, there are a number of estimators designed for filtering and smoothing of continous data. All estimators can be found within the estimators directory. The following estimators are currently implemented:
- Regression Splines
- Kernel Regression
- Kernel Regression with Robust Extensions (Median of Means)
- Segmented Regression 
- Trend Filtering
- Condtional Trend Filtering

All estimators inherit from the base class ```Base_Estimator```. This class provides the `fit` and `estimate` method. Estimators I am hoping to soon implement include:

- Variable Knot Splines
- Smoothing Splines