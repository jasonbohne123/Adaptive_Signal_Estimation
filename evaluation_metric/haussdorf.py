import numpy as np


def compute_haussdorf_distance(candidate_set, true_set):

    """
    Compute the Haussdorf distance between two sets of points

    Parameters
    ----------
    candidate_set : np.ndarray
        Candidate set of points
    true_set : np.ndarray
        True set of points

    Returns
    -------
    haussdorf_distance : float
        The Haussdorf distance between the two sets of points
    """

    # Compute the distance between each point in the candidate set and the true set
    distances = np.array([np.min(np.linalg.norm(candidate_set - true_point, axis=1)) for true_point in true_set])

    # Compute the distance between each point in the true set and the candidate set
    distances = np.append(
        distances,
        np.array([np.min(np.linalg.norm(true_set - candidate_point, axis=1)) for candidate_point in candidate_set]),
    )

    # Compute the Haussdorf distance
    haussdorf_distance = np.max(distances)

    return haussdorf_distance
