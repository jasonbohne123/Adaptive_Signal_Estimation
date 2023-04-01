### Simulation Infrastructure
- Base Simulator & Sampler for Random Sampling of a given distribution
- Conditional Simulator given certain occurrences within a prior
    - Local Max/Min 
    - Random Selected

 
### Integration with MLFlow
    - Runs of a particular model are stored in MLFlow sharing a common experiment
    - `mlflow_helpers` used to create mlflow experiment /log information
    - All MLFlow commands need to be ran from the corresponding directory `cd trend_filtering`
    - Visually inspect by navigating to mlflow folder `mflow ui`
    - Clear garbage collector with `mlflow gc`


### Design Separation
    - Models have their own wrappers to simulation class found in corresponding folder
    - Each model will have a file `run_bulk_{model_name}.py` to run a bulk simulation
    - Script wraps the model with simulation generation based on specific goal

 
