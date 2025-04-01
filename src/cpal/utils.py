"""
Utils for CPAL.
"""
import numpy as np
import cvxpy as cp
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

RANDOM_STATE = 0

def relu(x):
    return np.maximum(0,x)
def drelu(x):
    return x>=0
def sign(a):
    return 2 * int(a >= 0) - 1

def generate_hyperplane_arrangement(X,d = 2, P = 2000, seed = 0):
    # beta=1e-5
    n_train = len(X)
    np.random.seed(seed)
    dmat=np.empty((n_train,0))
    
    ## Finite approximation of all possible sign patterns
    for i in range(P):
        u=np.random.randn(d,1)
        dmat=np.append(dmat,drelu(np.dot(X,u)),axis=1)
    
    dmat=(np.unique(dmat,axis=1))

    return dmat

def solve_two_layer_convex_program(X, y, dmat, beta=1e-5, solver=cp.CLARABEL, verbose=False):
    """
    Solves the convex relaxation of a 2-layer ReLU network using mixed-norm regularization.

    Args:
        X: np.ndarray, shape (n, d), feature matrix
        y: np.ndarray, shape (n,), target vector
        dmat: np.ndarray, shape (n, m), binary matrix from the cutting-plane setup
        beta: float, regularization parameter
        solver: cvxpy solver to use (default: CLARABEL)
        verbose: bool, whether to print solver output

    Returns:
        Uopt1: np.ndarray, shape (d, m), optimal first-layer parameters
        Uopt2: np.ndarray, shape (d, m), optimal second-layer parameters
        obj_val: float, optimal objective value
        status: str, solver status
    """
    n, d = X.shape
    m = dmat.shape[1]
    
    # Variables
    Uopt1 = cp.Variable((d, m))
    Uopt2 = cp.Variable((d, m))

    # Predicted outputs
    yopt1 = cp.sum(cp.multiply(dmat, X @ Uopt1), axis=1)
    yopt2 = cp.sum(cp.multiply(dmat, X @ Uopt2), axis=1)
    
    # Objective
    residual = y - (yopt1 - yopt2)
    loss = cp.sum_squares(residual) / (2 * n)
    reg = beta * (cp.mixed_norm(Uopt1.T, 2, 1) + cp.mixed_norm(Uopt2.T, 2, 1))
    cost = loss + reg

    # Constraints
    constraint_mat = 2 * dmat - np.ones((n, m))
    constraints = [
        cp.multiply(constraint_mat, X @ Uopt1) >= 0,
        cp.multiply(constraint_mat, X @ Uopt2) >= 0
    ]

    # Solve
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=solver, warm_start=True, verbose=verbose)

    if prob.status != "optimal":
        print(f"Convex: Warning — problem status is {prob.status}")
    else:
        print(f"Convex: Optimal objective value = {prob.value:.6f}")

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

# plot evaluation for regression
def evaluate_quadratic_regression_models(X_all, y_all, X, y, X_test, y_test,
                                          Uopt1, Uopt2, U_bp, w_bp,
                                          save_path='full_data_reg.pdf',
                                          print_metrics=True,
                                          plot=True):
    """
    Evaluate and compare convex and BP models for y = x^2 regression.

    Args:
        X_all: full data (n, d+1) with bias
        y_all: full targets (n,)
        X: training data (n_train, d+1)
        y: training targets (n_train,)
        X_test: test data (n_test, d+1)
        y_test: test targets (n_test,)
        Uopt1, Uopt2: convex solution weight matrices (d, m)
        U_bp: learned input weights from BP (d, 2m)
        w_bp: learned output weights from BP (2m, 1)
        save_path: filename to save plot
        print_metrics: whether to print RMSE and R²
        plot: whether to show plot

    Returns:
        Dictionary of all RMSE and R² scores
    """
    # Predict full set
    yest_cvx = np.sum(drelu(X_all @ Uopt1) * (X_all @ Uopt1) -
                      drelu(X_all @ Uopt2) * (X_all @ Uopt2), axis=1)
    yest_bp = relu(X_all @ U_bp) @ w_bp
    yest_bp = yest_bp.flatten()

    # Predict train set
    yest_cvx_train = np.sum(drelu(X @ Uopt1) * (X @ Uopt1) -
                            drelu(X @ Uopt2) * (X @ Uopt2), axis=1)
    yest_bp_train = relu(X @ U_bp) @ w_bp
    yest_bp_train = yest_bp_train.flatten()

    # Predict test set
    yest_cvx_test = np.sum(drelu(X_test @ Uopt1) * (X_test @ Uopt1) -
                           drelu(X_test @ Uopt2) * (X_test @ Uopt2), axis=1)
    yest_bp_test = relu(X_test @ U_bp) @ w_bp
    yest_bp_test = yest_bp_test.flatten()

    # Ground truth: y = x^2
    x_vals = X_all[:, 0]
    y_true = x_vals ** 2

    # Metrics
    results = {
        'rmse_cvx': np.sqrt(mean_squared_error(y_true, yest_cvx)),
        'rmse_bp': np.sqrt(mean_squared_error(y_true, yest_bp)),
        'r2_cvx': r2_score(y_true, yest_cvx),
        'r2_bp': r2_score(y_true, yest_bp),
        'rmse_cvx_train': np.sqrt(mean_squared_error(y, yest_cvx_train)),
        'rmse_bp_train': np.sqrt(mean_squared_error(y, yest_bp_train)),
        'r2_cvx_train': r2_score(y, yest_cvx_train),
        'r2_bp_train': r2_score(y, yest_bp_train),
        'rmse_cvx_test': np.sqrt(mean_squared_error(y_test, yest_cvx_test)),
        'rmse_bp_test': np.sqrt(mean_squared_error(y_test, yest_bp_test)),
        'r2_cvx_test': r2_score(y_test, yest_cvx_test),
        'r2_bp_test': r2_score(y_test, yest_bp_test),
    }

    if print_metrics:
        print(f'Convex Optimization RMSE overall: {results["rmse_cvx"]:.4f}, R²: {results["r2_cvx"]:.4f}')
        print(f'Backpropagation RMSE overall:    {results["rmse_bp"]:.4f}, R²: {results["r2_bp"]:.4f}')
        print(f'Convex Optimization RMSE train:  {results["rmse_cvx_train"]:.4f}, R²: {results["r2_cvx_train"]:.4f}')
        print(f'Backpropagation RMSE train:      {results["rmse_bp_train"]:.4f}, R²: {results["r2_bp_train"]:.4f}')
        print(f'Convex Optimization RMSE test:   {results["rmse_cvx_test"]:.4f}, R²: {results["r2_cvx_test"]:.4f}')
        print(f'Backpropagation RMSE test:       {results["rmse_bp_test"]:.4f}, R²: {results["r2_bp_test"]:.4f}')

    if plot:
        sort_idx = np.argsort(x_vals)
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals[sort_idx], y_true[sort_idx], 'k--', label='True y = x²', linewidth=2)
        plt.plot(x_vals[sort_idx], yest_cvx[sort_idx], 'r-', label='Convex ReLU', linewidth=2)
        plt.plot(x_vals[sort_idx], yest_bp[sort_idx], 'c-', label='BP ReLU', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Quadratic Regression: Convex ReLU vs BP')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.savefig(save_path)
        plt.show()

    return results



# if __name__ == "__main__":
#     from synthetic_data import *
#     X_all, y_all, X, y, X_test, y_test = generate_quadratic_regression()
#     dmat = generate_hyperplane_arrangement(X = X, seed = RANDOM_STATE)
#     U1, U2, cvx_opt, status = solve_two_layer_convex_program(X, y, dmat, beta=1e-5)
#     print("Following are results from BP -----")
#     U_bp, w_bp, obj_bp = train_bp_two_layer_relu(
#     X=X, y=y, m=dmat.shape[1], beta=1e-5, sigma=1e-2, mu_init=0.5,
#     ITERS=15000, cvx_opt=cvx_opt, seed=42)
#     results = evaluate_quadratic_regression_models(
#         X_all=X_all, y_all=y_all,
#         X=X, y=y,
#         X_test=X_test, y_test=y_test,
#         Uopt1=U1, Uopt2=U2,
#         U_bp=U_bp, w_bp=w_bp,
#         save_path='full_data_reg.pdf'
#         )


