def get_segmentation_constants():

    return {
        "K_max": 0.025,  # good proxy is same as candidate changepoint threshold
        "order": 1,
        "cp_quantile": 0.025,  # good proxy is  1-2.5% of data length (25 for 1000)
        "min_cp_distance": 0.025,  # good proxy is 2.5-5% of data length (25-50 for 1000)
        "nu": 0.15,
    }
