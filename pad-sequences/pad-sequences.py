import numpy as np
import math
def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    lengths = [len(i) for i in seqs] 
    n = max(lengths)
    # seqs = np.array(seqs)
    if max_len!=None:
        n = max_len 
    seqs = [row[:n] for row in seqs]
    seqs = np.array([np.pad(row, (0, n - len(row)), constant_values=pad_value) for row in seqs])
    print(seqs)
    return seqs
    
    