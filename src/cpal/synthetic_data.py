"""
Scripts for generating synthetic test data, such as the binary spiral.
"""
import numpy as np
import matplotlib.pyplot as plt
import math

RANDOM_STATE = 0

# Spiral data generation functions
def spiral_xy(i, total_points, spiral_num, n_shape=50):
    i_normalized = i * n_shape / (total_points - 1)
    φ = i_normalized / 16 * math.pi
    r = 6.5 * ((104 - i_normalized) / 104)
    x = (r * math.cos(φ) * spiral_num) / 13 + 0.5
    y = (r * math.sin(φ) * spiral_num) / 13 + 0.5
    return (x, y)

def spiral(spiral_num, n=100, n_shape=50):
    return [spiral_xy(i, n, spiral_num, n_shape) for i in range(n // 2)]

def generate_spiral_data(n=100, n_train=80, n_shape=50, seed=RANDOM_STATE, default_label=True):
    a = spiral(1, n, n_shape)
    b = spiral(-1, n, n_shape)
    X_all = 2 * np.concatenate((a, b), axis=0) - 1
    X_all = np.append(X_all, np.ones((n, 1)), axis=1)  # Add bias term

    if default_label:
        y_all = np.concatenate((np.ones(n // 2), -np.ones(n // 2)))
    else:
        y_all = np.concatenate((np.ones(n // 2), np.zeros(n // 2)))

    np.random.seed(seed)
    idx = np.random.permutation(n)
    X_all, y_all = X_all[idx], y_all[idx]

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_test, y_test = X_all[n_train:], y_all[n_train:]
    return X_all, y_all, X_train, y_train, X_test, y_test

def plot_spiral_data(X_train, y_train, X_test=None, y_test=None, title='Spiral Data'):
    """
    Scatter plot of spiral classification data (train and optionally test).
    
    Args:
        X_train: np.ndarray of shape (n_train, 3), where last column is bias
        y_train: labels for training set, should be ±1
        X_test: (optional) test data, shape (n_test, 3)
        y_test: (optional) test labels, should be ±1
        title: plot title
    """
    # Remove bias column for plotting
    X_train = X_train[:, :2]
    if X_test is not None:
        X_test = X_test[:, :2]

    # Class-wise indexing
    train_pos = np.where(y_train == 1)
    train_neg = np.where(y_train == -1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X_train[train_pos, 0], X_train[train_pos, 1], c='c', label='Train +1', marker='o', edgecolors='k')
    ax.scatter(X_train[train_neg, 0], X_train[train_neg, 1], c='m', label='Train -1', marker='o', edgecolors='k')

    if X_test is not None and y_test is not None:
        test_pos = np.where(y_test == 1)
        test_neg = np.where(y_test == -1)
        ax.scatter(X_test[test_pos, 0], X_test[test_pos, 1], c='c', label='Test +1', marker='^', edgecolors='k')
        ax.scatter(X_test[test_neg, 0], X_test[test_neg, 1], c='m', label='Test -1', marker='^', edgecolors='k')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    #ax.grid(True)
    plt.show()


# generate simple regression data
def generate_quadratic_regression(n = 100, n_train = 80, seed = RANDOM_STATE, noise = False, noise_param = 0.05, plot = False):
    x_vals = np.linspace(-1, 1, n)  # Generate n points uniformly spaced between -1 and 1
    y_true = x_vals ** 2  # Calculate y = x^2
    
    # Add some noise to make it a more realistic regression problem
    if noise:
        np.random.seed(seed)
        noise = noise_param * np.random.randn(n)
        y_vals = y_true + noise
    else:
        y_vals = y_true
    
    # Plot the generated data
    if plot:
        plt.figure(figsize=(7, 7))
        plt.scatter(x_vals, y_vals, color='blue', label='Data Points')
        plt.plot(x_vals, y_true, color='black', label='True y = x^2', linewidth=2)
        
        plt.title('Continuous Quadratic Regression Dataset')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
    else:
        pass

    # Adding bias to the data
    X_all = np.column_stack((x_vals, np.ones(n)))  # Shape: (n, 2) where the second column is the bias term
    y_all = y_vals # y_all is unchanged

    # randomly select X,y
    np.random.seed(seed)
    
    # Randomly select 80 indices
    selected_indices = np.random.choice(np.arange(n), size=n_train, replace=False)
    unselected_indices = np.setdiff1d(np.arange(n), selected_indices)
    
    # Select the corresponding rows from X and values from Y for train dataset
    X = X_all[selected_indices]
    y = y_all[selected_indices]
    
    # Test set
    X_test = X_all[unselected_indices]
    y_test = y_all[unselected_indices]

    return X_all, y_all, X, y, X_test, y_test


# if __name__ == "__main__":
#     X_all, y_all, X_train, y_train, X_test, y_test = generate_spiral_data(n=100, n_train=80)
#     plot_spiral_data(X_train, y_train, X_test, y_test, title='Spiral Dataset: Train + Test')
