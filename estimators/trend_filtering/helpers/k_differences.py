def differences(seq, k=0):
    """Compute the kth-differences between elements of a sequence"""

    diff = []

    if k == 0:
        return seq

    for ct, element in enumerate(seq):

        # passes out the first k elements from the sequence
        if ct < k:
            continue

        # compute the kth difference
        diff.append(element - seq[ct - k])

    return diff
