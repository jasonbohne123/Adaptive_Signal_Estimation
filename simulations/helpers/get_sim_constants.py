def get_simulation_constants():
    """Default Simulation Constants for Trend Filtering"""

    constants = {
        "n_sims": 5,
        "n_samples": 1,
        # simulation params
        "underlying_dist": "normal",
        "label_style": "k_local_spikes",
        "signal_to_noise": 1,  # in time-sensitive case, this should be much larger
        "reference_variance": 1e-2,
        "shift": 100,
    }

    return constants
