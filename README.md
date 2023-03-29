### Adaptive Signal Estimation (Private Repo)

Collection of private algorithms related to topics for adaptive signal estimation

Work relevant to current research interests of conditional trend filtering and changepoint detection

Topics include

- Optimized numerical algorithms for sparse matrices and difference matrices
- Kernel Estimation Class with Robust Extensions (Median of Means)
- Simulation Class with Conditional Simulation extensions (Importance Sampling)
- Trend Filtering Algorithm via Primal Dual Optimization with adaptive extension



#### Installation

To replicate the enviroment used for this project, run the following commands:

Create a conda env using the environment.yml file

```conda env create -f environment.yml```

Setup Pre-commit Hooks for formatting

```conda install isort autoflake black pre-commit pre-commit install```


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


### Model Selection
