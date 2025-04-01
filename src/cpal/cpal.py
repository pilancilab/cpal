"""
Implements Cutting-Plane based active learning (CPAL).
"""
import cvxpy as cp
import numpy as np
from scipy.stats import norm
from utils import *
import warnings
warnings.filterwarnings('ignore')



def pred_point(i, U1v, U2v, X, dmat): # corresponds to <w(theta), x>
    y1 = np.sum(np.multiply(dmat[i],(X[i][np.newaxis, :] @ U1v)),axis=1)
    y2 = np.sum(np.multiply(dmat[i],(X[i][np.newaxis, :] @ U2v)),axis=1)
    return y1 - y2

def pred_point_simplified(i, U1v, U2v, X, dmat):
    var = np.vstack((U1v, U2v)).flatten(order='F')
    return (dmat[i] @ np.kron(np.eye(len(dmat[i])), np.concatenate((X[i], -X[i])).T)) @ var

def pred_point_simplified_vec(i, vec, X, dmat):
    return (dmat[i] @ np.kron(np.eye(len(dmat[i])), np.concatenate((X[i], -X[i])).T)) @ vec

def constraint(i, U1v, U2v, X, dmat):
    m=dmat.shape[1]
    return np.vstack((
        np.multiply((2*dmat[i]-np.ones((1,m))),(X[i] @ U1v)),
        np.multiply((2*dmat[i]-np.ones((1,m))),(X[i] @ U2v))
    )).flatten(order='F')

def constraint_simplified(i, U1v, U2v, X, dmat):
    
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

def center(S, dmat, d=2, R=1, boxinit=False, reg=False, beta =1e-5, reg_value=2e-4):
    """S is list of affine inequalities described as tuple of LHS vector/matrix and RHS scalar/vector"""
    m=dmat.shape[1]
    s = cp.Variable(2*d*m)
    obj = 0 if boxinit else cp.log(R - cp.norm(s))
    constraints = []
    if reg:
        U = cp.reshape(s, (2*d, m), order='F')
        #obj += cp.log(reg_value - beta*(cp.mixed_norm(U[:d].T,2,1)+cp.mixed_norm(U[d:].T,2,1)))
        constraints = [beta*(cp.mixed_norm(U[:d].T,2,1)+cp.mixed_norm(U[d:].T,2,1)) <= reg_value]
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
    return s.value

def cut(S, x, y, dmat_row, thresh):
    m = len(dmat_row)
    
    # Compute f^convex part
    f_convex = dmat_row @ np.kron(np.eye(m), np.concatenate((x, -x)).T)

    # Decompose the norm constraint into two linear constraints
    S.append((f_convex, y + thresh))  # First inequality: f_convex @ theta <= y + thresh
    S.append((-f_convex, -y + thresh))  # Second inequality: -f_convex @ theta <= -y + thresh

    # ReLU-based constraint
    relu_constraint = -np.kron(np.diag(2*dmat_row - np.ones(m)), np.kron(np.eye(2), x))
    for lhs in relu_constraint:
        S.append((lhs, 0))  # Append ReLU constraint to S

def query(C, c, data_tried, data_used, X, dmat, M=100):
    n_train, d = X.shape

    mini = np.inf
    i_mini = -1

    maxi = -np.inf
    i_maxi = -1

    minabs = np.inf
    i_minabs = -1

    for i in range(n_train): # search in finite data (D implicit) set to re-use dmat then
        if i not in data_tried and i not in data_used:
            pred = pred_point_simplified_vec(i, c, X, dmat)
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

# final convex solve

def convex_solve(used, d, dmat, n_train, beta = 1e-5):
    m=dmat.shape[1]
    used_unique = list(set(used))

    Uopt1_final=cp.Variable((d,m)) # c_i in paper
    Uopt2_final=cp.Variable((d,m)) # c_i' in paper

    yopt1_final=cp.sum(cp.multiply(dmat[used_unique],(X[used_unique]*Uopt1_final)),axis=1)
    yopt2_final=cp.sum(cp.multiply(dmat[used_unique],(X[used_unique]*Uopt2_final)),axis=1)

    cost=cp.sum_squares(y[used_unique]-(yopt1_final-yopt2_final))/(2*n_train)+beta*(cp.mixed_norm(Uopt1_final.T,2,1)+cp.mixed_norm(Uopt2_final.T,2,1))

    constraints=[]
    constraints+=[cp.multiply((2*dmat[used_unique]-np.ones((len(used_unique),m))),(X[used_unique]*Uopt1_final))>=0]
    constraints+=[cp.multiply((2*dmat[used_unique]-np.ones((len(used_unique),m))),(X[used_unique]*Uopt2_final))>=0]
    prob_final=cp.Problem(cp.Minimize(cost),constraints)
    prob_final.solve(solver=cp.CLARABEL,warm_start=True)

    return Uopt1_final.value, Uopt2_final.value, beta*(cp.mixed_norm(Uopt1_final.value.T,2,1)+cp.mixed_norm(Uopt2_final.value.T,2,1))

# cutting plane method

def cutting_plane_regression(X, y, dmat, n_points=100, maxit=10000, threshold = 1e-3, R = 1, boxinit=False):
    n_train, d = X.shape
    m=dmat.shape[1]
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
    while len(data_used) < n_points and it < maxit: # TODO: replace by proper termination criterion
        if len(data_tried) == n_train:
            data_tried = []
        if did_cut:
            #if len(data_used) > 0:
            #    _, _, reg_value = convex_solve(data_used)
            #else:
            #    reg_value = 1e-5
            c = center(Ct, R=R, dmat = dmat, d = d)
            did_cut = False
        i_mini, i_maxi, i_minabs = query(Ct, c, data_tried, data_used, X, dmat)
        if i_mini is None:
            return Ct, c, data_used
        data_tried += [i_mini, i_maxi]
        data_tried = list(set(data_tried))
        #if it >= 30:
        #    print(X[i_mini], np.sign(pred_point_simplified_vec(i_mini, c)), y[i_mini])
        #    print(X[i_maxi], np.sign(pred_point_simplified_vec(i_maxi, c)), y[i_maxi])
            #print(X[i_minabs], np.sign(pred_point_simplified_vec(i_minabs, c)), y[i_minabs])
        if True: #len(data_used) < 4 * n_points // 5:
            if np.linalg.norm(pred_point_simplified_vec(i_mini, c, X, dmat) - y[i_mini]) > threshold:
                print(f'Cutting at iteration {it}')
                #cut(Ct, X[i_mini], y[i_mini], dmat[i_mini])
                cut(Ct, X[i_mini], y[i_mini], dmat[i_mini],threshold)
                data_used.append(i_mini)
                did_cut = True
            if np.linalg.norm(pred_point_simplified_vec(i_maxi, c, X, dmat)- y[i_maxi]) > threshold:
                print(f'Cutting at iteration {it}')
                #cut(Ct, X[i_maxi], y[i_maxi], dmat[i_maxi])
                cut(Ct, X[i_maxi], y[i_maxi], dmat[i_maxi],threshold)
                data_used.append(i_maxi)
                did_cut = True
        else:
            if np.linalg.norm(pred_point_simplified_vec(i_minabs, c, X, dmat) - y[i_minabs]) > threshold:
                print(f'Cutting at iteration {it}')
                #cut(Ct, X[i_minabs], y[i_minabs], dmat[i_minabs])
                cut(Ct, X[i_minabs], y[i_minabs], dmat[i_minabs],threshold)
                data_used.append(i_minabs)
                did_cut = True
        it += 1

        #data_used = list(set(data_used))

        #print(len(data_tried))

    return Ct, c, data_used

# plotting functions
def visualize_regression(Uopt1v_list, Uopt2v_list, X_all, X, X_test, y_test, used, alpha = 0.95, plot_band = True, title = 'Quadratic Regression: True vs Predicted with Training Points'):
    
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

if __name__ == "__main__":
    from synthetic_data import *
    X_all, y_all, X, y, X_test, y_test = generate_quadratic_regression()
    dmat = generate_hyperplane_arrangement(X = X)
    C, c, used = cutting_plane_regression(X, y, dmat, n_points = 20)
    print(f'size of C: {len(C)}')
    print(f'used: {used}')
    # plot results
    n_train, d = X.shape
    m = dmat.shape[1]
    Uopt1_final_v, Uopt2_final_v, _ = convex_solve(used, d, dmat, n_train) # with final convex solve
    theta_matrix = np.reshape(c, (2*d, m), order='F') # without convex solve
    Uopt1_v = theta_matrix[:d]
    Uopt2_v = theta_matrix[d:]
    Uopt1v_list = [Uopt1_final_v, Uopt1_v]
    Uopt2v_list = [Uopt2_final_v, Uopt2_v]
    visualize_regression(Uopt1v_list, Uopt2v_list, X_all, X, X_test, y_test, used, plot_band = False)
