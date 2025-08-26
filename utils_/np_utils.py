import numpy as np



def np_sigmoid(x_in, z=1): 
    return np.clip( 1 / ( 1 + np.exp(-z*x_in) ), 0., 1.)


def dist_to_dist_kl_divergence(p, q):
    return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))


def kl_divergence_of_a_number_from_data_samples(sample: float, data_samples: np.ndarray, use_bins: int=200):
    
    counts, bins = np.histogram(data_samples, bins=use_bins)
    counts = counts.astype('float')
    counts[np.where(counts==0)] = 1e-8
    counts = counts / np.sum(counts)
    
    dist_ = np.random.normal(sample, scale=1e-3, size=50)
    
    _counts, _ = np.histogram(dist_, bins=bins)
    _counts = _counts.astype(np.float32)
    _counts[np.where(_counts==0)] = 1e-8
    _counts /= np.sum(_counts)
    
    return np.mean(dist_to_dist_kl_divergence(_counts, counts))

