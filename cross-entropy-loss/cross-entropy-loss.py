import numpy as np
from math import log
def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
   
    result = np.array(y_pred)
    y_true = np.array(y_true)
    N = y_true.size
    sum = 0 
    for i in range(N):
        sum = sum + np.log(result[i][y_true[i]])
        
    return -(1/N) * sum
    
        