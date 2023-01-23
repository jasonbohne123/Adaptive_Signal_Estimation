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

    # Check that the candidate set and the true set are not empty
    if len(candidate_set) == 0 or len(true_set) == 0:
        return np.inf

    # Compute the distance between each point in the candidate set and the true set
    distances = np.array([np.min(abs(candidate_set - true_point)) for true_point in true_set])

    # Compute the distance between each point in the true set and the candidate set
    distances = np.append(
        distances,
        np.array([np.min(abs(true_set - candidate_point)) for candidate_point in candidate_set]),
    )

    # Compute the Haussdorf distance
    haussdorf_distance = np.max(distances)

    return haussdorf_distance
