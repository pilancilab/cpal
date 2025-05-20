"""
Functions for plotting and evaluating the results of the CPAL algorithm.
"""

import numpy as np
from utils import relu, drelu
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# 1. Classification
# --------- Helper functions for plotting spiral decision boundary -----------------------
def generate_Xtest(samp=100, d=2):
    """
    Generates a grid of test input points over the square [-1, 1] x [-1, 1].

    Each point is represented as a d-dimensional vector, where the first two 
    dimensions correspond to the grid coordinates (x1, x2), and the remaining 
    dimensions (if any) are filled with 1s. By default, a bias term (1) is added 
    as the third component, so d should be at least 3 to avoid shape mismatch.

    Parameters:
    ----------
    samp : int, optional (default=100)
        Number of points sampled uniformly along each axis in the [-1, 1] interval.
        The total number of test points returned is samp^2.
    
    d : int, optional (default=2)
        The dimensionality of each test point. Must be at least 3 if including 
        a bias term as the third coordinate.

    Returns:
    -------
    Xtest : ndarray of shape (samp**2, d)
        A matrix of test input points laid out on a 2D grid in the first two 
        dimensions, with remaining dimensions (if any) set to 1. 
        Note: If d < 3, this function will raise a ValueError due to shape mismatch.
    """

    # Create 1D arrays of points from -1 to 1 for each axis
    x1 = np.linspace(-1, 1, samp).reshape(-1, 1)
    x2 = np.linspace(-1, 1, samp).reshape(-1, 1)

    # Initialize test point matrix with ones
    Xtest = np.ones((samp**2, d))

    count = 0
    for i in range(samp):
        for j in range(samp):
            # Fill in the first two dimensions with the grid coordinates
            # and set the third value to 1 (e.g., a bias term)
            Xtest[count] = [x1[i, 0], x2[j, 0], 1]
            count += 1

    return Xtest

# plotting
def plot_decision_boundary(X, y, X_test, y_test, Uopt1v, Uopt2v, selected_indices, name):
    # Define the grid range based on the data range
    x_min, x_max = -1.5, 1.5 # 1.5
    y_min, y_max = -1.5, 1.5

    # Create a grid of points
    x1 = np.linspace(x_min, x_max, 100)
    x2 = np.linspace(y_min, y_max, 100)
    x1, x2 = np.meshgrid(x1, x2)
    Xtest = np.c_[x1.ravel(), x2.ravel()]
    Xtest = np.append(Xtest, np.ones((Xtest.shape[0], 1)), axis=1)  # Add the bias term
    
    yest_cvx=np.sum(drelu(Xtest@Uopt1v)*(Xtest@Uopt1v)-drelu(Xtest@Uopt2v)*(Xtest@Uopt2v),axis=1)
    yest_cvx = yest_cvx.reshape(x1.shape)
    
    # Map labels back to -1 and 1 for visualization
    y_train_mapped = np.where(y == 1, 1, -1)
    y_test_mapped = np.where(y_test == 1, 1, -1)
    
    X_selected = X[selected_indices]
    y_selected = y_train_mapped[selected_indices]

    # Create subplots
    fig, ax = plt.subplots(figsize=(7, 7))

    # Define the custom colors
    colors = ['#920783', '#00b7c7']  # Switched the colors to match the image
    cmap = mcolors.ListedColormap(colors)

    # Plot the decision boundary with custom colors
    ax.contourf(x1, x2, yest_cvx, alpha=0.3, cmap=cmap)
    scatter_train = ax.scatter(X[:, 0], X[:, 1], c=y_train_mapped, edgecolor='k', s=20, cmap=cmap,
                               label='Train Data')
    scatter_test = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test_mapped, edgecolor='k', s=20, cmap=cmap,
                              marker='^', label = 'Test Data')
    scatter_select = ax.scatter(X_selected[:,0], X_selected[:,1], c=y_selected, s=80, cmap=cmap, marker='x',
                               label='Queried Data')
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f'{name}')
    plt.legend()
    #plt.savefig(f'{name}.pdf', bbox_inches='tight')
    plt.show()

# 2. Regression
# ----- Evaluation and plotting functions for regression -----

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

# plotting functions
def visualize_regression(Uopt1v_list, Uopt2v_list, X_all, X, y, X_test, y_test, used, alpha = 0.95, plot_band = True, title = 'Quadratic Regression: True vs Predicted with Training Points'):
    
    X_selected = X[used]
    y_selected = y[used]

    x_vals = X_all[:,:-1]
    
    # Plotting the true quadratic curve, predicted curves, and training points
    plt.figure(figsize=(8, 8))
    # Visualization and accuracy
    y_true = x_vals ** 2  # Since the true relationship is y = x^2
    # Plot the true curve y = x^2
    plt.plot(x_vals, y_true, 'k-', label='True y = x^2')
    plt.scatter(X_selected[:,:-1], y_selected, color='blue', s=50)
    plt.scatter(X_test[:,:-1], y_test, color='red', label='Test Data', alpha=0.5, marker='x')

    it = 0
    for it, (Uopt1v, Uopt2v) in enumerate(zip(Uopt1v_list, Uopt2v_list)):
        it += 1
        # overall result
        yest_cvx=np.sum(drelu(X_all@Uopt1v)*(X_all@Uopt1v)-drelu(X_all@Uopt2v)*(X_all@Uopt2v),axis=1)
        
        train_X_axis = X[:,:-1][:3].flatten() # for plotting purposes
        test_X_axis = X_test[:,:-1][:3].flatten()
        # train set result
        yest_cvx_train=np.sum(drelu(X@Uopt1v)*(X@Uopt1v)-drelu(X@Uopt2v)*(X@Uopt2v),axis=1)
        # test set result
        yest_cvx_test=np.sum(drelu(X_test@Uopt1v)*(X_test@Uopt1v)-drelu(X_test@Uopt2v)*(X_test@Uopt2v),axis=1)

        # Calculate RMSE for both convex optimization and backpropagation predictions
        rmse_cvx = np.sqrt(mean_squared_error(y_true, yest_cvx)) # overall
        rmse_cvx_train = np.sqrt(mean_squared_error(y, yest_cvx_train)) # train
        rmse_cvx_test = np.sqrt(mean_squared_error(y_test, yest_cvx_test)) # test
        
        # Calculate R^2 for both convex optimization and backpropagation predictions
        r2_cvx = r2_score(y_true, yest_cvx)
        r2_cvx_train = r2_score(y, yest_cvx_train)
        r2_cvx_test = r2_score(y_test, yest_cvx_test)

        if it == 1:
            label = 'Cutting-Plane (AFS)'
            # Plot the predicted curve from convex optimization
            plt.plot(x_vals, yest_cvx, color = 'red', label=f'Prediction ({label})', linewidth=2)
        else:
            label = 'Cutting-Plane (BFS)'
            # Plot the predicted curve from convex optimization
            plt.plot(x_vals, yest_cvx, label=f'Prediction ({label})', linewidth=2)
        
        # Print out the results
        print(label, f'RMSE overall: {rmse_cvx:.4f}, R^2: {r2_cvx:.4f}')
        print(label, f'RMSE over train set: {rmse_cvx_train:.4f}, R^2: {r2_cvx_train:.4f}')
        print(label, f'RMSE over test set: {rmse_cvx_test:.4f}, R^2: {r2_cvx_test:.4f}')
    

        # if plot band:
        if plot_band:
            # plot the alpha% confidence band
            residuals = y_true - yest_cvx
            std_error = np.std(residuals)
            z_value = norm.ppf(1 - (1 - alpha) / 2)
            # Calculate the confidence intervals
            upper_bound = yest_cvx + z_value * std_error
            lower_bound = yest_cvx - z_value * std_error
            #if line_color == 'magenta':
                #light_color = '#FFB3FF'
            #else:
                #light_color = 'lightcyan'
    
            plt.fill_between(x_vals, lower_bound, upper_bound, alpha=0.5, label=f'{int(alpha*100)}% Confidence Band')
    
    plt.title(f'Active Learning (Cutting-Plane)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'Cutting-Plane AFS.pdf', bbox_inches='tight')
    plt.show()