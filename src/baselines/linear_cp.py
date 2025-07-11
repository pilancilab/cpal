"""
Implements linear cutting-plane active learning by R&L for classification.
"""
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
from src.cpal.utils import *
from src.cpal.plot import *
import warnings
warnings.filterwarnings('ignore')


# Query function for linear cutting-plane active learning
def linear_query(w, X, y, data_tried, data_used):
    n_train, _ = X.shape

    mini = np.inf
    i_mini = -1

    maxi = -np.inf
    i_maxi = -1

    minabs = np.inf
    i_minabs = -1

    for i in range(n_train): 
        if i not in data_tried and i not in data_used:
            pred = y[i] * np.dot(w, X[i])  # linear prediction function
            if pred < mini:
                i_mini = 1*i
                mini = pred
            if pred > maxi:
                i_maxi = 1*i
                maxi = pred
            if abs(pred) < minabs:
                i_minabs = 1*i
                minabs = abs(pred)

    return i_mini, i_maxi, i_minabs


# compute analytic center
def linear_center(S, d, R=1, boxinit= False):
    """S is list of affine inequalities described as tuple of LHS vector/matrix and RHS scalar/vector"""
    s = cp.Variable(d)
    obj = 0 if boxinit else cp.log(R - cp.norm(s)) # objective for finding the center (log-barrier method)
    constraints = []
    if len(S) > 0:
        obj += cp.sum([cp.log(rhs - lhs @ s) for lhs, rhs in S])
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.MOSEK)
    if prob.status == cp.INFEASIBLE:
        print("The problem is infeasible.")
    # print(f'The objective value is {s.value}.')
    return s.value


def linear_cut_regression(S, x, y, thresh):
    # Decompose the norm constraint into two linear constraints
    S.append((x, y + thresh))  # First inequality: f_convex @ theta <= y + thresh
    S.append((-x, -y + thresh))  # Second inequality: -f_convex @ theta <= -y + thresh

def linear_cut_classification(S, x, y):
    """
    Add a new cutting-plane constraint for a linear classifier.
    
    Arguments:
    - S: current list of constraints
    - x: the feature vector of the queried point
    - y: the label of the queried point
    """
    # The constraint is simply: y * <w, x> >= 0, which means adding (y * x, 0) to the constraint set
    S.append((y * x, 0))  # Add new constraint y * <w, x> >= 0

# cutting-plane classification
def linear_cutting_plane_classification(X, y, m, n_points=100, maxit=10000, R = 1, boxinit=False):
    n_train, d = X.shape

    C0_lower = -R*np.ones(2*d*m)
    C0_upper = R*np.ones(2*d*m)

    data_tried = []
    data_used = []
    
    Ct = []
    if boxinit:
        for i, (l, u) in enumerate(zip(C0_lower, C0_upper)):
            one_vec = np.zeros(2*d*m)
            one_vec[i] = 1
            Ct.append((one_vec, u))
            Ct.append((-one_vec, -l))
    
    c = None
    did_cut = True
    it = 0
    while len(data_used) < n_points and it < maxit:
        if len(data_tried) == n_train:
            data_tried = []
        if did_cut:
            c = linear_center(Ct, d = d, R=R) # recompute center
            # Offset the center if it's too close to zero
            if np.all(np.abs(c) < 1e-6):  # If all values of `c` are very close to zero
                c += 1e-2 * np.random.randn(d) # Apply small random offset
            did_cut = False
        i_mini, i_maxi, _ = linear_query(c, X, y, data_tried, data_used)
        if i_mini is None:
            return Ct, c, data_used
        data_tried += [i_mini, i_maxi]
        data_tried = list(set(data_tried))
        if y[i_mini]*np.dot(c,X[i_mini]) < 0:
            linear_cut_classification(Ct, X[i_mini], y[i_mini])
            print(f'Cutting at iteration {it}')
            data_used.append(i_mini)
            did_cut = True
        if y[i_maxi]*np.dot(c,X[i_maxi]) < 0:
            linear_cut_classification(Ct, X[i_maxi], y[i_maxi])
            print(f'Cutting at iteration {it}')
            data_used.append(i_maxi)
            did_cut = True
        it += 1
    
    return Ct, c, data_used


def linear_cutting_plane_regression(X, y, n_points=100, maxit=10000, threshold = 1e-3, R = 1, boxinit=False):

    _, d = X.shape
    R = 1
    C0_lower_linear = -R*np.ones(d)
    C0_upper_linear = R*np.ones(d)

    data_tried = []
    data_used = []

    Ct = []
    if boxinit:
        for i, (l, u) in enumerate(zip(C0_lower_linear, C0_upper_linear)):
            one_vec = np.zeros(d)
            one_vec[i] = 1
            Ct.append((one_vec, u))
            Ct.append((-one_vec, -l))

    c = None
    did_cut = True
    it = 0
    #print(it)
    #print(len(data_used))
    while len(data_used) < n_points and it < maxit: 
        if did_cut:
            c = linear_center(Ct, d, R=R) # cannot be 0
            # Offset the center if it's too close to zero
            if np.all(np.abs(c) < 1e-2):  # If all values of `c` are very close to zero
                c += 1e-2 * np.random.randn(d)  # Apply small random offset
            print(c)
            did_cut = False
        i_mini, i_maxi, i_minabs = linear_query(c, X, y, data_tried, data_used)
        if i_mini is None:
            return Ct, c, data_used
        data_tried += [i_mini, i_maxi]
        data_tried = list(set(data_tried))
        if np.linalg.norm(y[i_mini]-np.dot(c,X[i_mini])) > threshold:
            print(f'Cutting at iteration {it}')
            linear_cut_regression(Ct, X[i_mini], y[i_mini], threshold)
            data_used.append(i_mini)
            did_cut = True
        if np.linalg.norm(y[i_maxi]-np.dot(c,X[i_maxi])) > threshold:
            print(f'Cutting at iteration {it}')
            linear_cut_regression(Ct, X[i_maxi], y[i_maxi], threshold)
            data_used.append(i_maxi)
            did_cut = True
        it += 1

    return Ct, c, data_used

### --------------- plotting functions for linear cutting plane active learning algorithm --------------- ###

def cal_linear_acc(X_train, y_train, X_test, y_test, itr_ls):
    acc_linear_train = []
    acc_linear_test = []
    for i in range(len(itr_ls)):
        _, c, _ = linear_cutting_plane_classification(itr_ls[i])
        y_pred_train = np.sign(np.dot(X_train, c))
        y_pred_test = np.sign(np.dot(X_test, c))
        accuracy_train = np.sum(y_pred_train == y_train) / len(y_train)
        accuracy_test = np.sum(y_pred_test == y_test) / len(y_test)
        acc_linear_train.append(accuracy_train)
        acc_linear_test.append(accuracy_test)
    return acc_linear_train, acc_linear_test

def plot_decision_boundary_linear(X_all, X_train, y_train, X_test, y_test, c, selected_indices, name = 'Decision Boundary (Linear Cutting-Plane)'):
    # Define the grid range based on the data range
    x_min, x_max = -1.5, 1.5 # 1.5
    y_min, y_max = -1.5, 1.5

    # Create a grid of points
    x1 = np.linspace(x_min, x_max, 100)
    x2 = np.linspace(y_min, y_max, 100)
    x1, x2 = np.meshgrid(x1, x2)
    Xtest = np.c_[x1.ravel(), x2.ravel()]
    Xtest = np.append(Xtest, np.ones((Xtest.shape[0], 1)), axis=1)  # Add the bias term
    
    # Predict the labels for both training and test data
    y_pred_train = np.sign(np.dot(X_train, c))  # Predictions for training data
    y_pred_test = np.sign(np.dot(X_test, c))  # Predictions for test data
    y_pred_all = np.sign(np.dot(X_all,c))
    y_pred_1 = np.sign(np.dot(Xtest,c))
    y_pred_1 = y_pred_1.reshape(x1.shape)
    
    # Map labels back to -1 and 1 for visualization
    y_train_mapped = np.where(y_train == 1, 1, -1)
    y_test_mapped = np.where(y_test == 1, 1, -1)

    # Compute accuracy on training and test sets
    accuracy_train = np.sum(y_pred_train == y_train) / len(y_train)
    accuracy_test = np.sum(y_pred_test == y_test) / len(y_test)

    print(f'Accuracy on training set: {accuracy_train * 100:.2f}%')
    print(f'Accuracy on test set: {accuracy_test * 100:.2f}%')
    
    X_selected = X_train[selected_indices]
    y_selected = y_train_mapped[selected_indices]

    # Create subplots
    fig, ax = plt.subplots(figsize=(7, 7))

    # Define the custom colors
    colors = ['#920783', '#00b7c7']  # Switched the colors to match the image
    cmap = mcolors.ListedColormap(colors)

    # Plot the decision boundary with custom colors
    ax.contourf(x1, x2, y_pred_1, alpha=0.3, cmap=cmap)
    scatter_train = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train_mapped, edgecolor='k', s=20, cmap=cmap,
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
    plt.savefig(f'{name}.pdf', bbox_inches='tight')
    plt.show()
 

# plot linear cpal for regression
def visualize_regression_linear(c, X_all, X, y, X_test, y_test, used, alpha = 0.95, plot_band = True):
    
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

    # overall result
    yest_linear=np.dot(X_all,c)
    
    train_X_axis = X[:,:-1][:3].flatten() # for plotting purposes
    test_X_axis = X_test[:,:-1][:3].flatten()
    # train set result
    yest_linear_train=np.dot(X,c)
    # test set result
    yest_linear_test=np.dot(X_test,c)

    # Calculate RMSE for both convex optimization and backpropagation predictions
    rmse_cvx = np.sqrt(mean_squared_error(y_true, yest_linear)) # overall
    rmse_cvx_train = np.sqrt(mean_squared_error(y, yest_linear_train)) # train
    rmse_cvx_test = np.sqrt(mean_squared_error(y_test, yest_linear_test)) # test
    
    # Calculate R^2 for both convex optimization and backpropagation predictions
    r2_cvx = r2_score(y_true, yest_linear)
    r2_cvx_train = r2_score(y, yest_linear_train)
    r2_cvx_test = r2_score(y_test, yest_linear_test)
    
    # Print out the results
    print(f'RMSE overall: {rmse_cvx:.4f}, R^2: {r2_cvx:.4f}')
    print(f'RMSE over train set: {rmse_cvx_train:.4f}, R^2: {r2_cvx_train:.4f}')
    print(f'RMSE over test set: {rmse_cvx_test:.4f}, R^2: {r2_cvx_test:.4f}')
    
    plt.plot(x_vals, yest_linear, label=f'Prediction (Linear)', linewidth=2)
    
    # if plot band:
    if plot_band:
        # plot the alpha% confidence band
        residuals = y_true - yest_linear
        std_error = np.std(residuals)
        z_value = norm.ppf(1 - (1 - alpha) / 2)
        # Calculate the confidence intervals
        upper_bound = yest_linear + z_value * std_error
        lower_bound = yest_linear - z_value * std_error

        plt.fill_between(x_vals, lower_bound, upper_bound, color='lightcyan', alpha=0.5, label=f'{int(alpha*100)}% Confidence Band')

    plt.title(f'Active Learning (Linear)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'plots/Linear.pdf', bbox_inches='tight')
    plt.show()

def visualize_regression_linear(c, X_all, X_train, y_train, X_test, y_test, used, alpha = 0.95, plot_band = True):
    
    X_selected = X_train[used]
    y_selected = y_train[used]

    x_vals = X_all[:, 0]
    
    # Plotting the true quadratic curve, predicted curves, and training points
    plt.figure(figsize=(8, 8))
    # Visualization and accuracy
    y_true = x_vals ** 2 
    # Plot the true curve y = x^2
    plt.plot(x_vals, y_true, 'k-', label='True y = x^2')
    plt.scatter(X_selected[:,:-1], y_selected, color='blue', s=50, label='Selected Data')
    plt.scatter(X_test[:,:-1], y_test, color='red', label='Test Data', alpha=0.5, marker='x')

    # overall result
    yest_linear=np.dot(X_all,c)

    # train set result
    yest_linear_train=np.dot(X_train,c)
    # test set result
    yest_linear_test=np.dot(X_test,c)

    # Calculate RMSE for both convex optimization and backpropagation predictions
    rmse_cvx = np.sqrt(mean_squared_error(y_true, yest_linear)) # overall
    rmse_cvx_train = np.sqrt(mean_squared_error(y_train, yest_linear_train)) # train
    rmse_cvx_test = np.sqrt(mean_squared_error(y_test, yest_linear_test)) # test
    
    # Calculate R^2 for both convex optimization and backpropagation predictions
    r2_cvx = r2_score(y_true, yest_linear)
    r2_cvx_train = r2_score(y_train, yest_linear_train)
    r2_cvx_test = r2_score(y_test, yest_linear_test)
    
    # Print out the results
    print(f'RMSE overall: {rmse_cvx:.4f}, R^2: {r2_cvx:.4f}')
    print(f'RMSE over train set: {rmse_cvx_train:.4f}, R^2: {r2_cvx_train:.4f}')
    print(f'RMSE over test set: {rmse_cvx_test:.4f}, R^2: {r2_cvx_test:.4f}')
    
    plt.plot(x_vals, yest_linear, label=f'Prediction (Linear)', linewidth=2)
    
    # if plot band:
    if plot_band:
        # plot the alpha% confidence band
        residuals = y_true - yest_linear
        std_error = np.std(residuals)
        z_value = norm.ppf(1 - (1 - alpha) / 2)
        # Calculate the confidence intervals
        upper_bound = yest_linear + z_value * std_error
        lower_bound = yest_linear - z_value * std_error

        plt.fill_between(x_vals, lower_bound, upper_bound, color='lightcyan', alpha=0.5, label=f'{int(alpha*100)}% Confidence Band')

    plt.title(f'Linear Cutting Plane on Quadratic Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'Linear_quadratic.pdf', bbox_inches='tight')
    plt.show()



