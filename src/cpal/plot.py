"""
Functions for plotting and evaluating the results of the CPAL algorithm.
"""

import numpy as np
from src.cpal.utils import relu, drelu
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.stats import norm

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
def plot_decision_boundary(
    X, y, X_test, y_test,
    Uopt1v=None, Uopt2v=None,
    U=None, w=None,
    selected_indices=None,
    name='Decision Boundary',
    plot_type='cvx'
):
    """
    Plots the decision boundary of a 2-layer ReLU network trained via either convex optimization or BP.

    Parameters
    ----------
    X : np.ndarray
        Training feature matrix of shape (n_samples, 2 or 3). Last column is bias if present.
    y : np.ndarray
        Training labels of shape (n_samples,). Should be +1 or -1.
    X_test : np.ndarray
        Test feature matrix of shape (n_test, 2 or 3). Last column is bias if present.
    y_test : np.ndarray
        Test labels of shape (n_test,). Should be +1 or -1.
    Uopt1v : np.ndarray, optional
        First-layer weight matrix from convex solution (d x m). Required if plot_type='cvx'.
    Uopt2v : np.ndarray, optional
        Second-layer weight matrix from convex solution (d x m). Required if plot_type='cvx'.
    U : np.ndarray, optional
        First-layer weight matrix from BP solution (d x 2m). Required if plot_type='bp'.
    w : np.ndarray, optional
        Second-layer weights from BP solution (2m x 1). Required if plot_type='bp'.
    selected_indices : np.ndarray, optional
        Indices of queried training points. If None, no queried points are highlighted.
    name : str
        Title of the plot and legend label.
    plot_type : str, default 'cvx'
        One of {'cvx', 'bp'}. Determines which model’s decision boundary to plot.

    Returns
    -------
    None
        Displays a matplotlib plot.
    """
    assert plot_type in ['cvx', 'bp'], "plot_type must be either 'cvx' or 'bp'"

    # Define grid for decision boundary
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    x1, x2 = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Xtest_grid = np.c_[x1.ravel(), x2.ravel()]
    Xtest_grid = np.append(Xtest_grid, np.ones((Xtest_grid.shape[0], 1)), axis=1)  # Add bias term

    # Compute predictions based on method
    if plot_type == 'cvx':
        assert Uopt1v is not None and Uopt2v is not None, "Uopt1v and Uopt2v are required for CVX plotting"
        scores = np.sum(
            drelu(Xtest_grid @ Uopt1v) * (Xtest_grid @ Uopt1v) -
            drelu(Xtest_grid @ Uopt2v) * (Xtest_grid @ Uopt2v),
            axis=1
        )
    else:  # 'bp'
        assert U is not None and w is not None, "U and w are required for BP plotting"
        scores = relu(Xtest_grid @ U) @ w
        scores = scores.flatten()

    Z = scores.reshape(x1.shape)

    # Normalize labels to {-1, 1}
    y_train_mapped = np.where(y == 1, 1, -1)
    y_test_mapped = np.where(y_test == 1, 1, -1)

    # Plotting setup
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ['#920783', '#00b7c7']  # magenta, cyan
    cmap = mcolors.ListedColormap(colors)

    # Plot decision region
    ax.contourf(x1, x2, Z, levels=0, alpha=0.3, cmap=cmap)

    # Plot training and test points
    ax.scatter(X[:, 0], X[:, 1], c=y_train_mapped, edgecolor='k', s=20, cmap=cmap, label='Train')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test_mapped, edgecolor='k', s=20, cmap=cmap, marker='^', label='Test')

    # Optionally plot queried points
    if selected_indices is not None:
        X_sel = X[selected_indices]
        y_sel = y_train_mapped[selected_indices]
        ax.scatter(X_sel[:, 0], X_sel[:, 1], c=y_sel, cmap=cmap, marker='x', s=80, label='Queried')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(name)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

# 2. Regression
# ----- Evaluation and plotting functions for regression -----

def evaluate_regression_models(X_all, y_true,
                                X_train, y_train,
                                X_test, y_test,
                                Uopt1, Uopt2,
                                U_bp=None, w_bp=None,
                                print_metrics=True,
                                plot=True,
                                name='Regression Comparison'):
    """
    Evaluate and compare convex and optional BP models for general regression.

    Parameters
    ----------
    X_all : np.ndarray
        Full data (n, d+1) with bias for prediction.
    y_true : np.ndarray
        True function values for X_all (same length as X_all).
    X : np.ndarray
        Training data (n_train, d+1).
    y : np.ndarray
        Training labels (n_train,).
    X_test : np.ndarray
        Test data (n_test, d+1).
    y_test : np.ndarray
        Test labels (n_test,).
    Uopt1, Uopt2 : np.ndarray
        Convex model weights (d, m).
    U_bp : np.ndarray, optional
        BP input weights (d, 2m).
    w_bp : np.ndarray, optional
        BP output weights (2m, 1).
    print_metrics : bool
        Whether to print RMSE and R² scores.
    plot : bool
        Whether to plot predicted curves.
    name : str
        Plot title.

    Returns
    -------
    dict
        Dictionary containing RMSE and R² scores for CVX and (if available) BP models.
    """
    # Convex predictions
    yest_cvx = np.sum(drelu(X_all @ Uopt1) * (X_all @ Uopt1) -
                      drelu(X_all @ Uopt2) * (X_all @ Uopt2), axis=1)
    yest_cvx_train = np.sum(drelu(X_train @ Uopt1) * (X_train @ Uopt1) -
                            drelu(X_train @ Uopt2) * (X_train @ Uopt2), axis=1)
    yest_cvx_test = np.sum(drelu(X_test @ Uopt1) * (X_test @ Uopt1) -
                           drelu(X_test @ Uopt2) * (X_test @ Uopt2), axis=1)

    results = {
        'rmse_cvx': np.sqrt(mean_squared_error(y_true, yest_cvx)),
        'r2_cvx': r2_score(y_true, yest_cvx),
        'rmse_cvx_train': np.sqrt(mean_squared_error(y_train, yest_cvx_train)),
        'r2_cvx_train': r2_score(y_train, yest_cvx_train),
        'rmse_cvx_test': np.sqrt(mean_squared_error(y_test, yest_cvx_test)),
        'r2_cvx_test': r2_score(y_test, yest_cvx_test),
    }

    # BP predictions (if provided)
    if U_bp is not None and w_bp is not None:
        yest_bp = relu(X_all @ U_bp) @ w_bp
        yest_bp = yest_bp.flatten()
        yest_bp_train = relu(X_train @ U_bp) @ w_bp
        yest_bp_train = yest_bp_train.flatten()
        yest_bp_test = relu(X_test @ U_bp) @ w_bp
        yest_bp_test = yest_bp_test.flatten()

        results.update({
            'rmse_bp': np.sqrt(mean_squared_error(y_true, yest_bp)),
            'r2_bp': r2_score(y_true, yest_bp),
            'rmse_bp_train': np.sqrt(mean_squared_error(y_train, yest_bp_train)),
            'r2_bp_train': r2_score(y_train, yest_bp_train),
            'rmse_bp_test': np.sqrt(mean_squared_error(y_test, yest_bp_test)),
            'r2_bp_test': r2_score(y_test, yest_bp_test),
        })

    if print_metrics:
        print(f'Convex RMSE full:  {results["rmse_cvx"]:.4f}, R²: {results["r2_cvx"]:.4f}')
        print(f'Convex RMSE train: {results["rmse_cvx_train"]:.4f}, R²: {results["r2_cvx_train"]:.4f}')
        print(f'Convex RMSE test:  {results["rmse_cvx_test"]:.4f}, R²: {results["r2_cvx_test"]:.4f}')
        if "rmse_bp" in results:
            print(f'BP RMSE full:      {results["rmse_bp"]:.4f}, R²: {results["r2_bp"]:.4f}')
            print(f'BP RMSE train:     {results["rmse_bp_train"]:.4f}, R²: {results["r2_bp_train"]:.4f}')
            print(f'BP RMSE test:      {results["rmse_bp_test"]:.4f}, R²: {results["r2_bp_test"]:.4f}')

    if plot:
        x_vals = X_all[:, 0]
        sort_idx = np.argsort(x_vals)
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals[sort_idx], y_true[sort_idx], 'k--', label='Ground Truth', linewidth=2)
        plt.plot(x_vals[sort_idx], yest_cvx[sort_idx], 'r-', label='Convex ReLU', linewidth=2)
        if "rmse_bp" in results:
            plt.plot(x_vals[sort_idx], yest_bp[sort_idx], 'c-', label='BP ReLU', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return results


def evaluate_classification_models(Xacc, yacc, Uopt1v, Uopt2v, U=None, w=None,
                                   X_train=None, y_train=None, verbose=True):
    """
    Compare classification performance between convex (CVX) and optionally backpropagation (BP) models.

    Parameters
    ----------
    Xacc : np.ndarray
        Test data matrix for accuracy evaluation (n_test, d).
    yacc : np.ndarray
        Ground-truth binary labels for test set (n_test,).
    Uopt1v : np.ndarray
        CVX first-layer weights of shape (d, m).
    Uopt2v : np.ndarray
        CVX second-layer weights of shape (d, m).
    U : np.ndarray, optional
        Backpropagation first-layer weights (d, 2m).
    w : np.ndarray, optional
        Backpropagation output weights (2m, 1).
    X_train : np.ndarray, optional
        Training input data (n_train, d).
    y_train : np.ndarray, optional
        Training binary labels.
    verbose : bool, optional
        If True, prints detailed evaluation results.

    Returns
    -------
    dict
        Dictionary with accuracy and count metrics for CVX (and optionally BP) on both train and test sets.
    """
    results = {}
    yacc = yacc.flatten()

    # --- CVX Test ---
    Z1_test = Xacc @ Uopt1v
    Z2_test = Xacc @ Uopt2v
    yhat_cvx_test = np.sum(drelu(Z1_test) * Z1_test - drelu(Z2_test) * Z2_test, axis=1)
    pred_cvx_test = np.sign(yhat_cvx_test)
    correct_cvx_test = np.sum(pred_cvx_test == yacc)
    acc_cvx_test = correct_cvx_test / len(yacc)

    results["cvx_correct_test"] = correct_cvx_test
    results["cvx_acc_test"] = acc_cvx_test

    # --- CVX Train (if available) ---
    if X_train is not None and y_train is not None:
        y_train = y_train.flatten()
        Z1_train = X_train @ Uopt1v
        Z2_train = X_train @ Uopt2v
        yhat_cvx_train = np.sum(drelu(Z1_train) * Z1_train - drelu(Z2_train) * Z2_train, axis=1)
        pred_cvx_train = np.sign(yhat_cvx_train)
        correct_cvx_train = np.sum(pred_cvx_train == y_train)
        acc_cvx_train = correct_cvx_train / len(y_train)

        results["cvx_correct_train"] = correct_cvx_train
        results["cvx_acc_train"] = acc_cvx_train

    # --- BP (if provided) ---
    if U is not None and w is not None:
        yhat_bp_test = relu(Xacc @ U) @ w
        pred_bp_test = np.sign(yhat_bp_test.flatten())
        correct_bp_test = np.sum(pred_bp_test == yacc)
        acc_bp_test = correct_bp_test / len(yacc)

        results["bp_correct_test"] = correct_bp_test
        results["bp_acc_test"] = acc_bp_test

        if X_train is not None and y_train is not None:
            yhat_bp_train = relu(X_train @ U) @ w
            pred_bp_train = np.sign(yhat_bp_train.flatten())
            correct_bp_train = np.sum(pred_bp_train == y_train)
            acc_bp_train = correct_bp_train / len(y_train)

            results["bp_correct_train"] = correct_bp_train
            results["bp_acc_train"] = acc_bp_train

    # --- Verbose Output ---
    if verbose:
        print("Convex Model (CVX):")
        print(f"  # correct on test set:  {correct_cvx_test}")
        print(f"  accuracy on test set:  {acc_cvx_test:.4f}")
        if "cvx_correct_train" in results:
            print(f"  # correct on train set: {correct_cvx_train}")
            print(f"  accuracy on train set: {acc_cvx_train:.4f}")
        if "bp_correct_test" in results:
            print("\nBackpropagation Model (BP):")
            print(f"  # correct on test set:  {correct_bp_test}")
            print(f"  accuracy on test set:  {acc_bp_test:.4f}")
            if "bp_correct_train" in results:
                print(f"  # correct on train set: {correct_bp_train}")
                print(f"  accuracy on train set: {acc_bp_train:.4f}")

    return results

def evaluate_model_performance(task,
                                X_all=None, y_true=None,
                                X_train=None, y_train=None,
                                X_test=None, y_test=None,
                                Uopt1=None, Uopt2=None,
                                U=None, w=None,
                                print_metrics=True,
                                plot=True,
                                name=None):
    """
    Unified evaluation wrapper for regression and classification tasks.

    Parameters
    ----------
    task : str
        One of {'r', 'c'}: 'r' for regression, 'c' for classification.
    X_all : np.ndarray, optional
        For regression: full input data (n, d+1) with bias.
    y_true : np.ndarray, optional
        For regression: true function values corresponding to X_all.
    X : np.ndarray, optional
        For regression: training data.
    y : np.ndarray, optional
        For regression: training targets.
    X_test : np.ndarray, optional
        For regression and classification: test data.
    y_test : np.ndarray, optional
        For regression and classification: test targets or labels.
    Uopt1, Uopt2 : np.ndarray
        Convex model weights.
    U, w : np.ndarray
        BP model weights.
    print_metrics : bool
        Whether to print metrics.
    plot : bool
        Whether to show regression plots.
    name : str, optional
        Title for regression plot.

    Returns
    -------
    dict or tuple
        Regression: dict of RMSE and R² scores.
        Classification: tuple of (accuracy_cvx, accuracy_bp).
    """
    if task == 'r':
        assert X_all is not None and y_true is not None and X_test is not None and y_test is not None
        return evaluate_regression_models(
            X_all=X_all,
            y_true=y_true,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            Uopt1=Uopt1,
            Uopt2=Uopt2,
            U_bp=U,
            w_bp=w,
            print_metrics=print_metrics,
            plot=plot,
            name=name or 'Regression Performance'
        )
    elif task == 'c':
        assert X_test is not None and y_test is not None
        return evaluate_classification_models(
            Xacc=X_test,
            yacc=y_test,
            Uopt1v=Uopt1,
            Uopt2v=Uopt2,
            U=U,
            w=w,
            X_train=X_train,
            y_train=y_train,
            verbose=print_metrics
        )
    else:
        raise ValueError(f"Unsupported task: {task}. Choose 'r' (regression) or 'c' (classification).")


# plotting functions
def visualize_quadratic_regression(Uopt1v_list, Uopt2v_list, X_all, X_train, y_train, X_test, y_test, used, alpha = 0.95, plot_band = True, title = 'Quadratic Regression: True vs Predicted with Training Points'):
    
    X_selected = X_train[used]
    y_selected = y_train[used]

    x_vals = X_all[:, 0]
    
    # Plotting the true quadratic curve, predicted curves, and training points
    plt.figure(figsize=(8, 8))
    # Visualization and accuracy
    y_true = x_vals ** 2 
    # Plot the true curve y = x^2
    plt.plot(x_vals, y_true, 'k-', label='True y = x^2')
    plt.scatter(X_selected[:,:-1], y_selected, label = 'Selected Data', color='blue', s=50)
    plt.scatter(X_test[:,:-1], y_test, color='red', label='Test Data', alpha=0.5, marker='x')

    it = 0
    for it, (Uopt1v, Uopt2v) in enumerate(zip(Uopt1v_list, Uopt2v_list)):
        it += 1
        # overall result
        yest_cvx=np.sum(drelu(X_all@Uopt1v)*(X_all@Uopt1v)-drelu(X_all@Uopt2v)*(X_all@Uopt2v),axis=1)
        
        # train set result
        yest_cvx_train=np.sum(drelu(X_train@Uopt1v)*(X_train@Uopt1v)-drelu(X_train@Uopt2v)*(X_train@Uopt2v),axis=1)
        # test set result
        yest_cvx_test=np.sum(drelu(X_test@Uopt1v)*(X_test@Uopt1v)-drelu(X_test@Uopt2v)*(X_test@Uopt2v),axis=1)

        # Calculate RMSE for both convex optimization and backpropagation predictions
        rmse_cvx = np.sqrt(mean_squared_error(y_true, yest_cvx)) # overall
        rmse_cvx_train = np.sqrt(mean_squared_error(y_train, yest_cvx_train)) # train
        rmse_cvx_test = np.sqrt(mean_squared_error(y_test, yest_cvx_test)) # test
        
        # Calculate R^2 for both convex optimization and backpropagation predictions
        r2_cvx = r2_score(y_true, yest_cvx)
        r2_cvx_train = r2_score(y_train, yest_cvx_train)
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
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(title, bbox_inches='tight')
    plt.show()