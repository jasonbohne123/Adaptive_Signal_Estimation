### Adaptive Signal Estimation (Public Repo)

Collection of estimators, basis functions, and other tools for adaptive signal estimation.

Organized into the following directories:

- estimators: Contains the estimators used for adaptive signal estimation
- basis_functions: Underlying basis functions used in regression 
- model_selection: Tools for hyperparameter tuning and model selection

Each directory contains a helpers directory that contains helper functions used in the directory. Constant files are across directories and contain constants used in the estimators.



#### Installation

To replicate the enviroment used for this project, run the following commands:

Create a conda env using the environment.yml file

```conda env create -f environment.yml```

Backup environment.yml file

```conda env export > environment.yml```

Setup Pre-commit Hooks for formatting

```conda install isort autoflake black pre-commit pre-commit install```
