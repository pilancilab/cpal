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
def safe_sign(x):
    s = np.sign(x)
    s[s == 0] = -1  # convert (0,1) to (-1,1)
    return s

def pred_point_simplified_vec(i, vec, X, dmat): # corresponds to <w(theta), x>
    return (dmat[i] @ np.kron(np.eye(len(dmat[i])), np.concatenate((X[i], -X[i])).T)) @ vec

def constraint(i, U1v, U2v, X, dmat):
    m=dmat.shape[1]
    return np.vstack((
        np.multiply((2*dmat[i]-np.ones((1,m))),(X[i] @ U1v)),
        np.multiply((2*dmat[i]-np.ones((1,m))),(X[i] @ U2v))
    )).flatten(order='F')

def constraint_simplified(i, U1v, U2v, X, dmat):
    m = dmat.shape[1]
    var = np.vstack((U1v, U2v)).flatten(order='F')
    return np.kron(np.diag(2*dmat[i]-np.ones(m)), np.kron(np.eye(2), X[i])) @ var

def sample_lattice(S, dmat, R=1):
    m=dmat.shape[1]
    d = 3
    l = cp.Variable(2*d*m)
    d = np.random.randn(2*d*m)
    obj = (d / np.linalg.norm(d)) @ l
    prob = cp.Problem(cp.Maximize(obj), [cp.norm(l) <= R] + [lhs @ l <= rhs for lhs, rhs in S])
    prob.solve(cp.MOSEK)
    return l.value


def sample_classifier(Ct, c, maxiter=10**5):
    for _ in range(maxiter):
        #candidate = np.random.uniform(C0_lower, C0_upper)
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

# helper function for plotting spiral decision boundary
def generate_Xtest(samp = 100, d = 2):
    x1=np.linspace(-1,1,samp).reshape(-1,1)
    x2=np.linspace(-1,1,samp).reshape(-1,1)
    Xtest=np.ones((samp**2,d))
    count=0
    for i in range(samp):
        for j in range(samp):
            Xtest[count]=[x1[i, 0],x2[j, 0],1]
            count+=1
    return Xtest

def generate_hyperplane_arrangement(X, P = 2000, seed = 0):
    n_train, d = X.shape
    np.random.seed(seed)
    dmat=np.empty((n_train,0))
    
    ## Finite approximation of all possible sign patterns
    for i in range(P):
        u=np.random.randn(d,1)
        dmat=np.append(dmat,drelu(np.dot(X,u)),axis=1)
    
    dmat=(np.unique(dmat,axis=1))

    return dmat

def solve_optimal_cvx(X, y, dmat, beta, solver=cp.CLARABEL, verbose=False): # taken from spiral
    """
    Solves the 2-layer convex optimization problem with group lasso penalty.

    Args:
        X (numpy.ndarray): Feature matrix of shape (n, d).
        y (numpy.ndarray): Labels vector of shape (n, 1).
        dmat (numpy.ndarray): Discretized activation matrix (n, m).
        beta (float): Regularization parameter for group lasso.
        solver (cvxpy Solver): Solver to use (default: CLARABEL).
        verbose (bool): If True, prints status and objective.

    Returns:
        float: Optimal objective value.
        tuple: (Uopt1, Uopt2) optimal variables as numpy arrays.
    """
    n, d = X.shape
    m = dmat.shape[1]

    Uopt1 = cp.Variable((d, m))  # First layer weight
    Uopt2 = cp.Variable((d, m))  # Second layer weight

    # Output predictions
    yopt1 = cp.sum(cp.multiply(dmat, X @ Uopt1), axis=1, keepdims=True)
    yopt2 = cp.sum(cp.multiply(dmat, X @ Uopt2), axis=1, keepdims=True)

    # Hinge-like squared loss + mixed norm regularization
    loss = cp.sum_squares(y - (yopt1 - yopt2)) / (2 * n)
    reg = beta * (cp.mixed_norm(Uopt1.T, 2, 1) + cp.mixed_norm(Uopt2.T, 2, 1))
    cost = loss + reg

    # Constraints to preserve sign of ReLU activations
    constraints = [
        cp.multiply((2 * dmat - 1), X @ Uopt1) >= 0,
        cp.multiply((2 * dmat - 1), X @ Uopt2) >= 0,
    ]

    # Solve
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=solver, warm_start=True)

    if verbose or prob.status != "optimal":
        print("Convex: Status:", prob.status)
        print("2-layer convex program objective value:", prob.value)

    return prob.value, Uopt1.value, Uopt2.value

def solve_two_layer_convex_program(X, y, dmat, beta=1e-5, solver=cp.CLARABEL, verbose=False): # taken from regression
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

def compare_accuracy(Xacc, yacc, Uopt1v, Uopt2v, U, w, verbose=True):
    """
    Compare accuracy between convex (CVX) model and backprop (BP) model.

    Args:
        Xacc (np.ndarray): Data matrix for accuracy evaluation.
        yacc (np.ndarray): Ground-truth binary labels (shape (n,) or (n,1)).
        Uopt1v (np.ndarray): CVX first-layer weights (d, m).
        Uopt2v (np.ndarray): CVX second-layer weights (d, m).
        U (np.ndarray): Backprop first-layer weights.
        w (np.ndarray): Backprop output layer weights.
        verbose (bool): If True, prints accuracy scores.

    Returns:
        tuple: (acc_cvx, acc_bp) — accuracy scores for CVX and BP models.
    """
    # Ensure labels are 1D
    yacc = yacc.flatten()

    # CVX prediction
    Z1 = Xacc @ Uopt1v
    Z2 = Xacc @ Uopt2v
    yhat_cvx = np.sum(drelu(Z1) * Z1 - drelu(Z2) * Z2, axis=1)
    yacc_cvx = np.sign(yhat_cvx)

    # BP prediction
    yhat_bp = relu(Xacc @ U) @ w
    yacc_bp = np.sign(yhat_bp.flatten())

    # Accuracy
    acc_cvx = np.mean(yacc_cvx == yacc)
    acc_bp = np.mean(yacc_bp == yacc)

    if verbose:
        print(f'Accuracy CVX: {acc_cvx:.4f}, Accuracy BP: {acc_bp:.4f}')

    return acc_cvx, acc_bp




# if __name__ == "__main__":
#     from synthetic_data import *
#     #X_all, y_all, X, y, X_test, y_test = generate_quadratic_regression()
#     X_all, y_all, X, y, X_test, y_test = generate_spiral_data(n=10, n_train=80)
#     dmat = generate_hyperplane_arrangement(X = X, P = 1000, seed = RANDOM_STATE)
#     U1, U2, cvx_opt, status = solve_two_layer_convex_program(X, y, dmat, beta=1e-5)
#     print("Following are results from BP -----")
#     U_bp, w_bp, obj_bp = train_bp_two_layer_relu(
#     X=X, y=y, m=dmat.shape[1], beta=1e-5, sigma=1e-2, mu_init=0.5,
#     ITERS=15000, cvx_opt=cvx_opt, seed=42)
    # results = evaluate_quadratic_regression_models(
    #     X_all=X_all, y_all=y_all,
    #     X=X, y=y,
    #     X_test=X_test, y_test=y_test,
    #     Uopt1=U1, Uopt2=U2,
    #     U_bp=U_bp, w_bp=w_bp,
    #     save_path='plots/full_data_classification.pdf'
    #     )


