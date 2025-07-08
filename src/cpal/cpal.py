"""
Implements Cutting-Plane based active learning (CPAL) for both classification and regression tasks.
"""
import cvxpy as cp
import numpy as np
from src.cpal.utils import *
from src.cpal.plot import *
import warnings
warnings.filterwarnings('ignore')


# compute analytic center
def center(S, X, dmat, R=1, boxinit=False, reg=False, beta =1e-5, reg_value=2e-4):
    """S is list of affine inequalities described as tuple of LHS vector/matrix and RHS scalar/vector"""
    _, d = X.shape
    m=dmat.shape[1]
    s = cp.Variable(2*d*m)
    obj = 0 if boxinit else cp.log(R - cp.norm(s))
    constraints = []
    if reg:
        U = cp.reshape(s, (2*d, m), order='F')
        constraints = [beta*(cp.mixed_norm(U[:d].T,2,1)+cp.mixed_norm(U[d:].T,2,1)) <= reg_value]
    if len(S) > 0:
        obj += cp.sum([cp.log(rhs - lhs @ s) for lhs, rhs in S])

    prob = cp.Problem(cp.Maximize(obj), constraints)

    solvers = ['MOSEK', 'CLARABEL', 'GLPK', 'SCS', 'ECOS', 'OSQP']
    for solver in solvers:
        try:
            prob.solve(solver=solver, verbose=False) # Warm_start set false for now.
            if prob.status == cp.OPTIMAL:
                print(f"Solver {solver} succeeded!")
                return s.value  # Return the center (concatenated c' and c vector)
        except Exception:
            if prob.status == cp.INFEASIBLE:
                print(f"Solver {solver} is infeasible.")
            else:
                pass
    
        raise RuntimeError("All solvers failed to find an optimal solution.")             

    return s.value

def cut_reg(S, x, y, dmat_row, thresh): # cut for reg-cpal
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


def cut_classification(S, x, y, dmat_row):
    m = len(dmat_row)
    S.append((-y * dmat_row @ np.kron(np.eye(m), np.concatenate((x, -x)).T), 0))

    relu_constraint = -np.kron(np.diag(2*dmat_row-np.ones(m)), np.kron(np.eye(2), x))
    for lhs in relu_constraint:
        S.append((lhs, 0))


def query(c, data_tried, data_used, X, dmat):
    n_train, _ = X.shape

    mini = np.inf
    i_mini = -1

    maxi = -np.inf
    i_maxi = -1

    minabs = np.inf
    i_minabs = -1

    for i in range(n_train): # search in finite data (D implicit) set to re-use dmat
        if i not in data_tried and i not in data_used:
            pred = pred_point(i, c, X, dmat)
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
def convex_solve(used, X, y, dmat, beta = 1e-5):
    n_train, d = X.shape
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


# cutting plane method for regression
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
    while len(data_used) < n_points and it < maxit:
        if len(data_tried) == n_train:
            data_tried = []
        if did_cut:
            c = center(Ct, X=X, R=R, dmat = dmat)
            did_cut = False
        i_mini, i_maxi, i_minabs = query(c, data_tried, data_used, X, dmat)
        if i_mini is None:
            return Ct, c, data_used
        data_tried += [i_mini, i_maxi]
        data_tried = list(set(data_tried))
        if np.linalg.norm(pred_point(i_mini, c, X, dmat) - y[i_mini]) > threshold:
            print(f'Cutting at iteration {it}')
            cut_reg(Ct, X[i_mini], y[i_mini], dmat[i_mini],threshold)
            data_used.append(i_mini)
            did_cut = True
        if np.linalg.norm(pred_point(i_maxi, c, X, dmat)- y[i_maxi]) > threshold:
            print(f'Cutting at iteration {it}')
            cut_reg(Ct, X[i_maxi], y[i_maxi], dmat[i_maxi],threshold)
            data_used.append(i_maxi)
            did_cut = True
        it += 1

    return Ct, c, data_used

# cutting-plane classification
def cutting_plane_classification(X, y, dmat, n_points=100, maxit=10000, R = 1, boxinit=False):
    _, d = X.shape
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

    while len(data_used) < n_points and it < maxit:
        if did_cut:
            c = center(Ct, dmat = dmat, X=X, R=R) # recompute center
            did_cut = False
        i_mini, i_maxi, _ = query(c, data_tried, data_used, X, dmat)
        if i_mini is None:
            return Ct, c, data_used
        data_tried += [i_mini, i_maxi]
        data_tried = list(set(data_tried))
        if np.sign(pred_point(i_mini, c, X, dmat)) != y[i_mini]:
            print(f'Cutting at iteration {it}')
            cut_classification(Ct, X[i_mini], y[i_mini], dmat[i_mini])
            data_used.append(i_mini)
            did_cut = True
        if np.sign(pred_point(i_maxi, c, X, dmat)) != y[i_maxi]:
            print(f'Cutting at iteration {it}')
            cut_classification(Ct, X[i_maxi], y[i_maxi], dmat[i_maxi])
            data_used.append(i_maxi)
            did_cut = True
        it += 1
    
    return Ct, c, data_used

