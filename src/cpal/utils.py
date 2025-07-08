"""
Utils for the main CPAL algorithm.
"""
import numpy as np


RANDOM_STATE = 0

# --- Activation and Sign Utilities ---

def relu(x):
    """
    Applies the ReLU (Rectified Linear Unit) function element-wise.
    
    Parameters:
    x : np.ndarray
        Input array.
    
    Returns:
    np.ndarray
        Element-wise max(x, 0).
    """
    return np.maximum(0, x)

def drelu(x):
    """
    Returns the derivative (or binary activation pattern) of the ReLU function.
    Specifically, returns True where x >= 0, False otherwise.
    
    Parameters:
    x : np.ndarray
        Input array.
    
    Returns:
    np.ndarray (boolean)
        Element-wise mask of where x >= 0.
    """
    return x >= 0

def sign(a):
    """
    Returns the sign of a scalar a, with output in {-1, 1}.
    
    Parameters:
    a : float
        A scalar value.
    
    Returns:
    int
        1 if a >= 0, else -1.
    """
    return 2 * int(a >= 0) - 1

def safe_sign(x):
    """
    Applies np.sign but converts all 0s to -1, so output values are only in {-1, 1}.
    
    Parameters:
    x : np.ndarray
        Input array.
    
    Returns:
    np.ndarray
        Array of -1 or 1, with 0s converted to -1.
    """
    s = np.sign(x)
    s[s == 0] = -1
    return s

# --- Main Function: Hyperplane Arrangement Generator ---

def generate_hyperplane_arrangement(X, P=2000, seed=0):
    """
    Generates a finite approximation of hyperplane-induced binary activation patterns (sign patterns)
    from a ReLU network using randomly sampled hyperplanes.

    Each hyperplane corresponds to a ReLU activation unit. For each hyperplane, the binary vector
    indicating whether the dot product of each point with the hyperplane normal is positive
    (i.e., ReLU activated) is computed and recorded.

    Parameters:
    X : np.ndarray of shape (n_train, d)
        Input data with n_train samples, each of dimension d.
    P : int
        Number of random hyperplanes to sample (default: 2000).
    seed : int
        Random seed for reproducibility (default: 0).
    
    Returns:
    dmat : np.ndarray of shape (n_train, K)
        Matrix of unique binary activation patterns (each column corresponds to a sampled hyperplane),
        where K <= P is the number of unique activation patterns.
    """
    n_train, d = X.shape
    np.random.seed(seed)
    dmat = np.empty((n_train, 0), dtype=bool)

    # Finite approximation of sign patterns induced by ReLU units
    for _ in range(P):
        u = np.random.randn(d, 1)  # Sample random hyperplane direction
        pattern = drelu(np.dot(X, u))  # Binary ReLU activation pattern
        dmat = np.concatenate((dmat, pattern), axis=1)  # Append pattern as new column

    # Remove duplicate columns (i.e., duplicate patterns)
    dmat = np.unique(dmat, axis=1)

    return dmat

# --- Prediction and Constraint Functions ---
def pred_point(i, vec, X, dmat): # corresponds to <w(theta), x>
    return (dmat[i] @ np.kron(np.eye(len(dmat[i])), np.concatenate((X[i], -X[i])).T)) @ vec

def constraint(i, U1v, U2v, X, dmat):
    m = dmat.shape[1]
    var = np.vstack((U1v, U2v)).flatten(order='F')
    return np.kron(np.diag(2*dmat[i]-np.ones(m)), np.kron(np.eye(2), X[i])) @ var

# sampling function
def sample_classifier(Ct, c, maxiter=10**5):
    for _ in range(maxiter):
        candidate = c + np.random.randn(*c.shape)
        if in_Ct(candidate, Ct):
            return candidate
    print(f'Failed to sample after {maxiter} tries.')
    return None

def in_Ct(c, Ct, eps=1e-3):
    for lhs, rhs in Ct:
        if lhs @ c > rhs + eps:
            return False
    return True


