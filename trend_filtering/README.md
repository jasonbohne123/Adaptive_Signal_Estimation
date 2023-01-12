### L1 Trend Filtering

Big Idea: Determine Linear Trend in a time series. Can be used to identify changepoints. Extension involves adapting sensitivity to noise across time

### File Structure
- Primal-Dual Optimization can be found within `adaptive_tf.py`
- Script to test problem with embedded cross validation `test_adaptive_tf.py`
- Constants for model and simulations found within `tf_constants.py`
- Wrapper for simulations found within `run_bulk_tf.py` 
- Various other helpers and research


### Ideas for Simulation Design
- Examine performance across a range of SNR values (sample_varaince)
- Across a grid of grid sizes. How important is a large candidate grid in CV
- Across a range of system sizes
