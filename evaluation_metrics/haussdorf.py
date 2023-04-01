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

    # each set always is closed on both ends (bound haussdorf distance by len(y)/2)
    padded_candidate_set = np.unique(np.concatenate([[0], candidate_set, [len(true_set)]]))
    padded_true_set = np.unique(np.concatenate([[0], true_set, [len(candidate_set)]]))

    # Compute the distance between each point in the candidate set and the true set
    distances = np.array([np.min(abs(padded_candidate_set - true_point)) for true_point in padded_true_set])

    # Compute the distance between each point in the true set and the candidate set
    distances = np.append(
        distances,
        np.array([np.min(abs(padded_true_set - candidate_point)) for candidate_point in padded_candidate_set]),
    )

    # Compute the Haussdorf distance
    haussdorf_distance = np.max(distances)

    return haussdorf_distance
