"""
Functions for solving the two-layer convex neural network (CVXNN) problem using CVXPY.
"""
from utils import relu, drelu, RANDOM_STATE
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

    return Uopt1.value, Uopt2.value, prob.value, prob.status


def train_bp_two_layer_relu(X, y, m, beta=1e-5, sigma=1e-2, mu_init=0.5, ITERS=15000, decay_step=8000, decay_factor=5, seed=RANDOM_STATE, plot=True, cvx_opt=None):
    """
    Trains a 2-layer ReLU network using vanilla gradient descent (BP-style).
    
    Args:
        X: np.ndarray, shape (n, d), input features
        y: np.ndarray, shape (n,), target values (e.g., ±1 for classification)
        m: int, number of neurons used in convex model → BP uses 2m neurons
        beta: float, regularization strength
        sigma: float, standard deviation for weight initialization
        mu_init: float, initial learning rate
        ITERS: int, number of gradient descent steps
        decay_step: int, how often to decay the learning rate
        decay_factor: float, learning rate decay factor
        seed: int or None, for reproducible initialization
        plot: bool, whether to plot the objective over iterations
        cvx_opt: float or None, optional baseline line for plotting

    Returns:
        U: learned input layer weights (d x 2m)
        w: learned second layer weights (2m x 1)
        obj_bp: array of objective values over time
    """
    np.random.seed(seed)
    
    n_train, d = X.shape
    mbp = 2 * m  # Double width for BP
    y = y.reshape(-1, 1)
    
    # Initialize weights
    U = sigma * np.random.randn(d, mbp)
    w = sigma * np.random.randn(mbp, 1)

    # Storage for objective
    obj_bp = np.empty((ITERS, 1))
    mu = mu_init

    for i in range(ITERS):
        # Decay learning rate
        if i % decay_step == 0 and i > 0:
            mu = mu / decay_factor

        # Full-batch GD
        a1 = X @ U
        yest = relu(a1) @ w
        yest_all = relu(X @ U) @ w  # full prediction

        # Objective function (MSE + L2 regularization)
        loss = np.linalg.norm(y - yest_all) ** 2 / (2 * n_train)
        reg = beta / 2 * (np.linalg.norm(U, 'fro') ** 2 + np.linalg.norm(w, 'fro') ** 2)
        obj_bp[i] = loss + reg

        # Gradients
        gradw = relu(a1).T @ (yest - y) / n_train
        gradU = X.T @ (drelu(a1) * ((yest - y) @ w.T)) / n_train

        # Gradient descent update
        U = (1 - mu * beta) * U - mu * gradU
        w = (1 - mu * beta) * w - mu * gradw

    # Optional plot
    if plot:
        plt.figure(figsize=(6, 4))
        plt.semilogy(obj_bp, label='BP GD')
        if cvx_opt is not None:
            plt.axhline(cvx_opt, color='k', linestyle='--', label='CVX Opt')
        plt.xlabel('Iteration')
        plt.ylabel('Objective')
        plt.title('Training Objective (BP)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return U, w, obj_bp