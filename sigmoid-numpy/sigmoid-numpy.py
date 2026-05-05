import numpy as np
import math
def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.array(x)
    return (1/(1+ np.exp(-x)))