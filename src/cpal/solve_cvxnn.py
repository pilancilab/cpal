"""
Functions for solving the two-layer convex neural network (CVXNN) problem using CVXPY.
"""
from src.cpal.utils import relu, drelu, RANDOM_STATE
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp


# --------- 2-layer convex nn solver -----------------------

def solve_two_layer_cvx(X, y, dmat, beta=1e-5, task_type='c', solver=cp.CLARABEL, verbose=False):
    """
    Solves the convex relaxation of a 2-layer ReLU network using group Lasso regularization.
    
    Supports both classification ('c') and regression ('r') tasks.
    
    Args:
        X (np.ndarray): Feature matrix of shape (n, d).
        y (np.ndarray): Target vector, shape (n,) for regression or (n, 1) for classification.
        dmat (np.ndarray): Binary discretized ReLU activation matrix of shape (n, m).
        beta (float): Regularization parameter.
        task_type (str): 'c' for classification or 'r' for regression.
        solver (cvxpy Solver): CVXPY solver (default: CLARABEL).
        verbose (bool): If True, print solver status and objective value.

    Returns:
        Uopt1 (np.ndarray): First-layer weights of shape (d, m).
        Uopt2 (np.ndarray): Second-layer weights of shape (d, m).
        obj_val (float): Optimal objective value.
        status (str): Solver status.
    """
    n, d = X.shape
    m = dmat.shape[1]

    # Handle label vector shape
    if task_type == 'c':
        if y.ndim == 1:
            y = y.reshape(-1, 1)  # Ensure (n, 1)
    elif task_type == 'r':
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)  # Flatten to (n,)
    else:
        raise ValueError("task_type must be 'c' (classification) or 'r' (regression)")

    # Variables
    Uopt1 = cp.Variable((d, m))
    Uopt2 = cp.Variable((d, m))

    # Compute prediction terms
    if task_type == 'c':
        yopt1 = cp.sum(cp.multiply(dmat, X @ Uopt1), axis=1, keepdims=True)
        yopt2 = cp.sum(cp.multiply(dmat, X @ Uopt2), axis=1, keepdims=True)
        residual = y - (yopt1 - yopt2)
    else:  # 'r'
        yopt1 = cp.sum(cp.multiply(dmat, X @ Uopt1), axis=1)
        yopt2 = cp.sum(cp.multiply(dmat, X @ Uopt2), axis=1)
        residual = y - (yopt1 - yopt2)

    # Objective: loss + group lasso regularization
    loss = cp.sum_squares(residual) / (2 * n)
    reg = beta * (cp.mixed_norm(Uopt1.T, 2, 1) + cp.mixed_norm(Uopt2.T, 2, 1))
    cost = loss + reg

    # ReLU sign constraints
    constraint_mat = 2 * dmat - np.ones((n, m))
    constraints = [
        cp.multiply(constraint_mat, X @ Uopt1) >= 0,
        cp.multiply(constraint_mat, X @ Uopt2) >= 0
    ]

    # Solve the problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=solver, warm_start=True, verbose=verbose)

    if verbose or prob.status != "optimal":
        print(f"[{task_type.upper()}] Convex program status: {prob.status}")
        print(f"[{task_type.upper()}] Objective value: {prob.value:.6f}")

    print("2-layer convex program objective value: ", prob.value)

    return Uopt1.value, Uopt2.value, prob.value, prob.status

# --------- 2-layer BP-style ReLU network training -----------------------

def train_bp_two_layer_relu(
    X, y, m, beta=1e-5, sigma=1e-2, mu_init=0.5,
    ITERS=15000, decay_step=8000, decay_factor=5,
    seed=42, plot=True, cvx_opt=None
):
    """
    Trains a 2-layer fully connected ReLU neural network using vanilla gradient descent,
    matching the behavior of a standard backpropagation (BP) routine.

    This implementation uses 2m hidden units (twice the number used in the corresponding
    convex relaxation) and includes L2 regularization on both layers. The learning rate
    decays by a fixed factor after a set number of iterations. Objective values are tracked
    and optionally plotted against a convex baseline.

    Args:
        X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target output vector of shape (n_samples,). Should be reshaped
                        to column vector internally for MSE loss computation.
        m (int): Number of hidden neurons used in the convex formulation. This BP model uses 2m.
        beta (float): L2 regularization strength for both layers.
        sigma (float): Standard deviation used for Gaussian initialization of weights.
        mu_init (float): Initial learning rate for gradient descent.
        ITERS (int): Total number of gradient descent iterations.
        decay_step (int): Number of steps before decaying the learning rate.
        decay_factor (float): Factor by which to divide the learning rate at each decay step.
        seed (int): Random seed for reproducible weight initialization and sampling.
        plot (bool): Whether to generate and display a semilog plot of training objective.
        cvx_opt (float or None): Optional reference value (e.g., from convex training) to
                                 plot as a horizontal line for comparison.

    Returns:
        U (np.ndarray): Learned first-layer weights of shape (n_features, 2m).
        w (np.ndarray): Learned second-layer weights of shape (2m, 1).
        obj_bp (np.ndarray): Array of shape (ITERS, 1) containing the objective value at each iteration.

    Notes:
        - The loss function optimized is the mean squared error (MSE) with L2 regularization:
              L = (1/2n) * ||y - f(X)||^2 + (beta/2)(||U||^2 + ||w||^2)
          where f(X) = ReLU(X @ U) @ w.
        - Weight updates are performed using full-batch gradient descent with learning rate decay.
        - The implementation can be adapted to mini-batch SGD if needed.
    """
    np.random.seed(seed)

    n, d = X.shape
    mbp = 2 * m
    mu = mu_init
    yall = y.reshape(-1, 1)

    # Initialize weights
    U = sigma * np.random.randn(d, mbp)
    w = sigma * np.random.randn(mbp, 1)
    obj_bp = np.empty((ITERS, 1))

    batch_size = n
    for i in range(ITERS):
        if i % decay_step == 0 and i > 0:
            mu = mu / decay_factor

        # Mini-batch sample (actually full-batch here)
        samp = np.random.choice(n, batch_size)
        Xgd = X[samp, :]
        ygd = yall[samp, :]

        a1 = Xgd @ U
        yest = relu(a1) @ w
        yest_all = relu(X @ U) @ w

        loss = np.linalg.norm(yall - yest_all)**2 / (2 * n)
        reg = (beta / 2) * (np.linalg.norm(U, 'fro')**2 + np.linalg.norm(w, 'fro')**2)
        obj_bp[i] = loss + reg

        gradw = relu(a1).T @ (yest - ygd) / batch_size
        gradU = Xgd.T @ (drelu(a1) * ((yest - ygd) @ w.T)) / batch_size

        U = (1 - mu * beta) * U - mu * gradU
        w = (1 - mu * beta) * w - mu * gradw

    # Plot
    if plot:
        plt.figure(figsize=(6, 4))
        plt.semilogy(obj_bp, label='GD')
        if cvx_opt is not None:
            plt.axhline(cvx_opt, color='k', linestyle='--', label='Optimal')
        plt.xlabel('Iteration')
        plt.ylabel('Objective value')
        plt.title('BP vs CVX')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('bp_vs_cvx.pdf', bbox_inches='tight')
        plt.show()

    print("2-layer BP objective value: ", obj_bp[-1, 0])
    return U, w, obj_bp