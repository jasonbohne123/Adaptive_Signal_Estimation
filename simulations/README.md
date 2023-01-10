### Simulation Infrastructure

1. Partition Data and Prep for Simulations

2. Run Simulations (Under Various Conditions)
    - Original Cross-Validated Param
    - Conditional(Adaptive) Cross-Validated Param
    - Across Various times of day, days of week, etc.

3. Auto-Analyze Simulations
    - Analyze Statistics and Results

4. Store and Save Results 
    - Runs of a particular model are stored in MLFlow sharing a common experiment
    - All MLFlow commands need to be ran from the corresponding directory `cd trend_filtering`
    - Visually inspect by navigating to mlflow folder `mflow ui`
    - Delete experiements or runs `mlflow gc`

 
