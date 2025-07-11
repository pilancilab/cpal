"""
Implements AL baselines from scikit-learn.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from skactiveml.pool import GreedySamplingX, GreedySamplingTarget, QueryByCommittee, \
    KLDivergenceMaximization
from sklearn.ensemble import BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error
from skactiveml.utils import MISSING_LABEL

from skactiveml.regressor import NICKernelRegressor, SklearnRegressor
from skactiveml.utils import call_func, is_labeled
from scipy.stats import norm, uniform
import warnings
warnings.filterwarnings('ignore')

mlp.rcParams["figure.facecolor"] = "white"

# helper function for adapting data to fit the scikit-learn framework - i.e. removing the bias column
def adapt_data_for_scikit_al(X_all, X, X_test):
    X_all = X_all[:,:-1]
    X = X[:,:-1]
    X_test = X_test[:,:-1]
    return X_all, X, X_test


# Active Learning Function (X_train is 1d!)
def active_learning_skactiveml(X_train, y_train, data_budget=10, strategy='greedyX', seed=42):
    # Initialize the regressor
    reg = NICKernelRegressor(metric_dict={'gamma': 15.0})

    np.random.seed(seed)

    X_selected = []
    y_selected = []

    # X_pool is the remaining data points from X_train without bias (initial pool)
    X_pool = X_train.reshape(-1, 1)  # Ensure X_train is one-dimensional
    y_pool = y_train
    y_pool_query = np.full_like(y_pool, np.nan)  # Important for active learning

    selected_indices = []

    # Define the active learning strategy
    if strategy == 'greedyX':
        al_strategy = GreedySamplingX(random_state=seed)
    elif strategy == 'greedyTarget':
        al_strategy = GreedySamplingTarget(random_state=seed)
    elif strategy == 'qbc':
        al_strategy = QueryByCommittee(random_state=seed)
    elif strategy == 'kldiv':
        al_strategy = KLDivergenceMaximization(
            random_state=seed,
            integration_dict_target_val={
                "method": "assume_linear",
                "n_integration_samples": 3,
            },
            integration_dict_cross_entropy={
                "method": "assume_linear",
                "n_integration_samples": 3,
            }
        )
    else:
        raise ValueError("Invalid strategy selected.")

    # Query counter
    query_count = 0
    while query_count < data_budget:
        # Use the AL strategy to query the next index
        reg.fit(X_pool, y_pool_query)
        
        query_idx, utils = call_func(al_strategy.query,
                X=X_pool,
                y=y_pool_query,
                reg=reg,
                ensemble=SklearnRegressor(BaggingRegressor(reg, n_estimators=4)),
                fit_reg=True,
                return_utilities=True,
            )

        query_idx = query_idx[0]

        y_pool_query[query_idx] = y_pool[query_idx]

        selected_indices.append(query_idx)

        # Update the base regressor with the selected points
        # reg.fit(X_selected, y_selected.ravel())

        X_selected.append(X_train[query_idx])
        y_selected.append(y_train[query_idx])
        query_count += 1
        
    # Return the final selected dataset
    return X_selected, y_selected, selected_indices, reg

def run_active_learning_strategies(X_all, y_all, X, y, X_test, y_test,
                                   strategies, active_learning_fn,
                                   data_budget=20, seed=0,
                                   save_plots=True, show_plots=True):
    """
    Runs multiple active learning strategies and evaluates regression performance.

    Args:
        X_all: np.ndarray, shape (n, d), full input dataset
        y_all: np.ndarray, shape (n,), full targets
        X: np.ndarray, shape (n_train, d), training inputs
        y: np.ndarray, shape (n_train,), training targets
        X_test: np.ndarray, shape (n_test, d), test inputs
        y_test: np.ndarray, shape (n_test,), test targets
        strategies: list of str, strategy names to run (e.g., ['greedyX', 'qbc'])
        active_learning_fn: function to call: (X, y, data_budget, seed, strategy) → (X_selected, y_selected, _, reg)
        data_budget: int, number of samples to query
        seed: int, random seed
        save_plots: bool, whether to save plots as PDFs
        show_plots: bool, whether to display plots

    Returns:
        results: dict mapping strategy name → dict with predictions and RMSEs
    """
    results = {}

    for strategy in strategies:
        print(f"Running strategy: {strategy}")
        X_selected, y_selected, _, reg = active_learning_fn(X, y, data_budget=data_budget, seed=seed, strategy=strategy)

        # Predict
        y_pred_test = reg.predict(X_test)
        y_pred_train = reg.predict(X)
        y_pred_all = reg.predict(X_all)

        # Compute metrics
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        rmse_train = np.sqrt(mean_squared_error(y, y_pred_train))
        rmse_all = np.sqrt(mean_squared_error(y_all, y_pred_all))

        results[strategy] = {
            'X_selected': X_selected,
            'y_selected': y_selected,
            'y_pred': y_pred_all,
            'rmse_test': rmse_test,
            'rmse_train': rmse_train,
            'rmse_overall': rmse_all
        }

        # Plot
        if save_plots or show_plots:
            plt.figure(figsize=(8, 6))
            plt.plot(X_all.ravel(), y_all, 'k-', label='True y = x²')
            plt.scatter(X_selected, y_selected, color='blue', label='Queried Points', s=50)
            plt.scatter(X_test, y_test, color='red', label='Test Data', alpha=0.5, marker='x')
            plt.plot(X_all.ravel(), y_pred_all, label=f'Prediction ({strategy})', linewidth=2)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Active Learning ({strategy})')
            plt.legend()
            if save_plots:
                plt.savefig(f'plots/{strategy}.pdf', bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    # Print RMSE for each strategy
    for strategy, result in results.items():
        print(f'{strategy} Test RMSE: {result["rmse_test"]:.4f}; Train RMSE: {result["rmse_train"]:.4f}; Overall RMSE: {result["rmse_overall"]:.4f}')

    return results

