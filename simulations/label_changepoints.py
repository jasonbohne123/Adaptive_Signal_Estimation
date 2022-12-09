
import numpy as np
from scipy.signal import find_peaks

def label_changepoints (prior, cp_label_style,k_points):
    """Returns indices of true changepoints across a given prior from prespecified criteria"""

    if cp_label_style == 'trivial':
        cp_index = np.arange(len(prior))

    elif cp_label_style == 'k_maxima':
        cp_index=np.argpartition(prior,-k_points)[-k_points:]

    elif cp_label_style == 'k_minima':
        cp_index=np.argpartition(prior,k_points)[:k_points]
    
    elif cp_label_style == 'k_local_spikes':
        cp_index, _ = find_peaks(prior, height=0, distance=1)

    elif cp_label_style == 'k_local_valleys':
        cp_index, _ = find_peaks(-prior, height=0, distance=1)

    else:
        raise ValueError('Invalid changepoint labeling style')
    
    
    concat_index = np.unique(np.concatenate((np.array([0]),cp_index,np.array([len(prior)])-1)))
    return sorted(concat_index)




   