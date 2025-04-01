"""
Implements various evaluation of baselines vs. our method. Includes:
(1). error-bar plots
(2). deep-nn training results using samples selected by each method
"""
import os
from src.deep_nn import SimpleRegressor, train_model
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# helper functions
def format_selected_data(X, y, selected_indices, method_name = 'cpal'):
    """
    Formats selected indices into a results-style dictionary under key 'cpal'.

    Args:
        X: np.ndarray, shape (n, d), full feature matrix with bias column
        y: np.ndarray, shape (n,), target vector
        selected_indices: list or np.ndarray of ints
        method_name : str, name of the method

    Returns:
        dict in format:
        {
            'cpal': {
                'X_selected': [array([x1]), array([x2]), ...],
                'y_selected': array([y1, y2, ...])
            }
        }
    """
    selected_indices = np.asarray(selected_indices).astype(int)
    # Ensure selected Xs are reshaped as individual arrays (like AL format)
    X_selected = [np.array([x[0]]) for x in X[selected_indices]]  # take only feature, not bias
    y_selected = y[selected_indices]

    return {
        method_name: {
            'X_selected': X_selected,
            'y_selected': y_selected
        }
    }


# regression
def evaluate_dnn_from_al_results_regression(results, X_all, y_all, noise=False, epochs=1000, lr=0.01, save_dir='plots'):
    """
    Trains and evaluates a deep neural network regressor on AL-selected data
    from multiple strategies and plots the prediction results.
    """
    os.makedirs(save_dir, exist_ok=True)
    x_plot = X_all[:, 0]
    y_true = x_plot**2
    X_all_tensor = torch.tensor(X_all, dtype=torch.float32)

    grand_results = {}  # to store predictions + selected points per strategy

    for strategy, data in results.items():
        print(f"\nTraining DNN on strategy: {strategy}")
        X_sel_raw = data['X_selected']
        y_sel = np.array(data['y_selected'])

        # Convert X_selected from list-of-arrays → 2D array and add bias
        X_sel = np.vstack(X_sel_raw)
        X_sel_with_bias = np.hstack([X_sel, np.ones((X_sel.shape[0], 1))])

        # Convert to tensors
        X_tensor = torch.tensor(X_sel_with_bias, dtype=torch.float32)
        y_tensor = torch.tensor(y_sel, dtype=torch.float32).view(-1, 1)

        # Train DNN
        model = SimpleRegressor()
        trained_model = train_model(model, X_tensor, y_tensor, epochs=epochs, lr=lr)

        # Predict on full data
        trained_model.eval()
        with torch.no_grad():
            y_pred = trained_model(X_all_tensor).numpy().flatten()

        # Sort for plotting
        sort_idx = np.argsort(x_plot)
        grand_results[strategy] = {
            'x_sorted': x_plot[sort_idx],
            'y_pred_sorted': y_pred[sort_idx],
            'X_sel': X_sel.flatten(),
            'y_sel': y_sel
        }

        # Individual plot
        plt.figure(figsize=(8, 5))
        plt.plot(x_plot[sort_idx], y_true[sort_idx], 'k--', label='True Function')
        plt.plot(x_plot[sort_idx], y_pred[sort_idx], 'r-', label='DNN Prediction', linewidth=2)
        if noise:
            plt.scatter(X_sel.flatten(), y_sel, color='blue', label='Queried (noisy)', s=50)
        plt.title(f'DNN Fit using AL Strategy: {strategy}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        filename = os.path.join(save_dir, f'dnn_{strategy}.pdf')
        plt.savefig(filename, bbox_inches='tight')
        plt.show()

    # Grand plot with all strategies
    plt.figure(figsize=(10, 6))
    plt.plot(np.sort(x_plot), np.sort(x_plot)**2, 'k--', label='True Function', linewidth=2)

    colors = cm.tab10.colors
    color_idx = 0

    for strategy, data in grand_results.items():
        if strategy == 'cpal':
            plt.plot(data['x_sorted'], data['y_pred_sorted'], label='cpal (ours)', color='red', linewidth=2)
        else:
            color = colors[color_idx % len(colors)]
            plt.plot(data['x_sorted'], data['y_pred_sorted'], label=strategy, color=color, linewidth=2)
            color_idx += 1

    if noise:
        for data in grand_results.values():
            plt.scatter(data['X_sel'], data['y_sel'], color='blue', s=20, alpha=0.5)

    plt.title('DNN Predictions Across AL Strategies')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(save_dir, 'dnn_all_strategies_comparison.pdf')
    plt.savefig(filename, bbox_inches='tight')
    plt.show()