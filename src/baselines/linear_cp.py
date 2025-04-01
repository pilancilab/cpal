"""
Implements linear cutting-plane active learning by R&L.

# TODO: debug - something is not right
"""
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm


# compute analytic center
def linear_center(S, d, R=1, boxinit= False):
    """S is list of affine inequalities described as tuple of LHS vector/matrix and RHS scalar/vector"""
    s = cp.Variable(d)
    obj = 0 if boxinit else cp.log(R - cp.norm(s))
    constraints = []
    if len(S) > 0:
        obj += cp.sum([cp.log(rhs - lhs @ s) for lhs, rhs in S])

    # solvers = ['MOSEK', 'CLARABEL', 'GLPK', 'SCS', 'ECOS', 'OSQP']
    # for solver in solvers:
    #     try:
    #         prob.solve(solver=solver, warm_start = True, verbose=False)
    #         if prob.status == cp.OPTIMAL:
    #             print(f"Solver {solver} succeeded!")
    #             return s.value  # Return the center (concatenated c' and c vector)
    #     except Exception:
    #         if prob.status == cp.INFEASIBLE:
    #             print(f"Solver {solver} is infeasible.")
    #         else:
    #             pass
    
    # raise RuntimeError("All solvers failed to find an optimal solution.")             
    
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.MOSEK)
    if prob.status == cp.INFEASIBLE:
        print("The problem is infeasible.")
    print(s.value)
    return s.value

def linear_query(w, X, y, data_tried, data_used, M=100):
    n_train, d= X.shape
    mini = np.inf
    i_mini = -1

    maxi = -np.inf
    i_maxi = -1

    minabs = np.inf
    i_minabs = -1
    
    for i in range(n_train): # search in finite data (D implicit) set to re-use dmat then
        if i not in data_tried and i not in data_used:
            pred = y[i] * np.dot(w, X[i])
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

def linear_cut(S, x, y, thresh):
    # Decompose the norm constraint into two linear constraints
    S.append((x, y + thresh))  # First inequality: f_convex @ theta <= y + thresh
    S.append((-x, -y + thresh))  # Second inequality: -f_convex @ theta <= -y + thresh

def linear_cutting_plane(X, y, n_points=100, maxit=10000, threshold = 1e-3, R = 1, boxinit=False):

    n_train, d = X.shape
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
        if len(data_tried) == n_train:
            data_tried = []
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
        if True: 
            if np.linalg.norm(y[i_mini]-np.dot(c,X[i_mini])) > threshold:
                #print(1,'y')
                #print(f'Cutting at iteration {it}')
                linear_cut(Ct, X[i_mini], y[i_mini], threshold)
                data_used.append(i_mini)
                did_cut = True
            if np.linalg.norm(y[i_maxi]-np.dot(c,X[i_maxi])) > threshold:
                #print(2,'y')
                #print(f'Cutting at iteration {it}')
                linear_cut(Ct, X[i_maxi], y[i_maxi], threshold)
                data_used.append(i_maxi)
                did_cut = True
        else:
            if np.linalg.norm(y[i_minabs]*np.dot(c,X[i_minabs])) > threshold:
                #print(3,'y')
                #print(f'Cutting at iteration {it}')
                linear_cut(Ct, X[i_minabs], y[i_minabs], threshold)
                data_used.append(i_minabs)
                did_cut = True
        it += 1

        #data_used = list(set(data_used))

        #print(len(data_tried))

    return Ct, c, data_used


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


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from src.cpal.synthetic_data import *
    X_all, y_all, X, y, X_test, y_test = generate_quadratic_regression(seed = RANDOM_STATE, plot = False)

    C, c, used = linear_cutting_plane(X = X, y=y, n_points = 4)
    print(f'size of C: {len(C)}')
    print(f'used: {used}')

    visualize_regression_linear(c, X_all, X, y, X_test, y_test, used, alpha = 0.95, plot_band = False)
