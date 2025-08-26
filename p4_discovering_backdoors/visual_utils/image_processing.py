import numpy as np



def normalize(x):
    return (x-np.min(x)) / (np.max(x)-np.min(x))

