"""
Temporary codes here for cpal in classification - will clean up and unify the codes later.
"""
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from src.cpal.synthetic_data import generate_spiral_data
import matplotlib.colors as mcolors
from src.cpal.utils import *
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
    m = dmat.shape[1]
    return np.vstack((
        np.multiply((2*dmat[i]-np.ones((1,m))),(X[i] @ U1v)),
        np.multiply((2*dmat[i]-np.ones((1,m))),(X[i] @ U2v))
    )).flatten(order='F')

def constraint_simplified(i, U1v, U2v, X, dmat):
    m = dmat.shape[1]
    var = np.vstack((U1v, U2v)).flatten(order='F')
    return np.kron(np.diag(2*dmat[i]-np.ones(m)), np.kron(np.eye(2), X[i])) @ var

def in_Ct(c, Ct, eps=1e-3):
    for lhs, rhs in Ct:
        if lhs @ c > rhs + eps:
            return False
    return True

def sample_lattice(dmat, S, R=1):
    m = dmat.shape[1]
    d = 3
    l = cp.Variable(2*d*m)
    d = np.random.randn(2*d*m)
    obj = (d / np.linalg.norm(d)) @ l
    prob = cp.Problem(cp.Maximize(obj), [cp.norm(l) <= R] + [lhs @ l <= rhs for lhs, rhs in S])
    prob.solve(cp.MOSEK)
    return l.value

def sample_classifier(Ct, c, maxiter=10**5):
    for _ in range(maxiter):
        candidate = c + np.random.randn(*c.shape)
        if in_Ct(candidate, Ct):
            return candidate
    print(f'Failed to sample after {maxiter} tries.')
    return None

def query(C, c, X, dmat, data_tried, data_used, M=100):

    n_train, d = X.shape

    mini = np.inf
    i_mini = -1

    maxi = -np.inf
    i_maxi = -1

    minabs = np.inf
    i_minabs = -1

    #theta_s = np.zeros(2*d*m)
    #for _ in range(M):
    #    th = sample_classifier(C, c)
    #    #th = sample_lattice(C)
    #    if th is None:
    #        return None, None, None
    #    theta_s += (1/M) * th

    #theta_matrix = np.reshape(c, (2*d, m), order='F')
    #U1_query=theta_matrix[:d]
    #U2_query=theta_matrix[d:]

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

# compute analytic center
def center(X, dmat, S, R=1, boxinit=False, reg=False, beta = 1e-5, reg_value=2e-4):
    """S is list of affine inequalities described as tuple of LHS vector/matrix and RHS scalar/vector"""
    n_train, d = X.shape
    m = dmat.shape[1]
    s = cp.Variable(2*d*m)
    obj = 0 if boxinit else cp.log(R - cp.norm(s))
    constraints = []
    if reg:
        U = cp.reshape(s, (2*d, m), order='F')
        #obj += cp.log(reg_value - beta*(cp.mixed_norm(U[:d].T,2,1)+cp.mixed_norm(U[d:].T,2,1)))
        constraints = [beta*(cp.mixed_norm(U[:d].T,2,1)+cp.mixed_norm(U[d:].T,2,1)) <= reg_value]
    if len(S) > 0:
        obj += cp.sum([cp.log(rhs - lhs @ s) for lhs, rhs in S])
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.MOSEK)
    return s.value

# cut set
def cut(S, x, y, dmat_row):
    m = len(dmat_row)
    # f = cp.sum(cp.multiply(dmat, X*(Uopt1 - Uopt2)), axis=1)
    # f_i = dmat[i, 0] * X[i, :] [I - I] var[:, 0] + dmat[i, 1] * X[i, :] [I - I] var[:, 1] + ...  where var = [Uopt1; Uopt2]
    # f_i = dmat[i, :] @ [X[i, :] [I - I] var[:, 0]; X[i, :] [I - I] var[:, 1]; ...]
    S.append((-y * dmat_row @ np.kron(np.eye(m), np.concatenate((x, -x)).T), 0))

    relu_constraint = -np.kron(np.diag(2*dmat_row-np.ones(m)), np.kron(np.eye(2), x))
    for lhs in relu_constraint:
        S.append((lhs, 0))

def convex_solve(used, X, y, dmat, beta = 1e-5):

    n, d = X.shape
    m = dmat.shape[1]

    used_unique = list(set(used))

    Uopt1_final=cp.Variable((d,m)) # c_i in paper
    Uopt2_final=cp.Variable((d,m)) # c_i' in paper

    yopt1_final=cp.sum(cp.multiply(dmat[used_unique],(X[used_unique]*Uopt1_final)),axis=1)
    yopt2_final=cp.sum(cp.multiply(dmat[used_unique],(X[used_unique]*Uopt2_final)),axis=1)

    cost=cp.sum_squares(y[used_unique]-(yopt1_final-yopt2_final))/(2*n)+beta*(cp.mixed_norm(Uopt1_final.T,2,1)+cp.mixed_norm(Uopt2_final.T,2,1))

    constraints=[]
    constraints+=[cp.multiply((2*dmat[used_unique]-np.ones((len(used_unique),m))),(X[used_unique]*Uopt1_final))>=0]
    constraints+=[cp.multiply((2*dmat[used_unique]-np.ones((len(used_unique),m))),(X[used_unique]*Uopt2_final))>=0]
    prob_final=cp.Problem(cp.Minimize(cost),constraints)
    prob_final.solve(solver=cp.CLARABEL,warm_start=True)

    return Uopt1_final.value, Uopt2_final.value, beta*(cp.mixed_norm(Uopt1_final.value.T,2,1)+cp.mixed_norm(Uopt2_final.value.T,2,1))

# cutting plane method

def cutting_plane(X, y, dmat, n_points=100, maxit=10000, R = 1, boxinit=False):

    n_train, d = X.shape
    m = dmat.shape[1]

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
            c = center(X, dmat, Ct, R=R)
            did_cut = False
        i_mini, i_maxi, i_minabs = query(Ct, c, X, dmat, data_tried, data_used)
        if i_mini is None:
            return Ct, c, data_used
        data_tried += [i_mini, i_maxi]
        data_tried = list(set(data_tried))
        #if it >= 30:
        #    print(X[i_mini], np.sign(pred_point_simplified_vec(i_mini, c)), y[i_mini])
        #    print(X[i_maxi], np.sign(pred_point_simplified_vec(i_maxi, c)), y[i_maxi])
            #print(X[i_minabs], np.sign(pred_point_simplified_vec(i_minabs, c)), y[i_minabs])
        if True: #len(data_used) < 4 * n_points // 5:
            if np.sign(pred_point_simplified_vec(i_mini, c, X, dmat)) != y[i_mini]:
                print(f'Cutting at iteration {it}')
                cut(Ct, X[i_mini], y[i_mini], dmat[i_mini])
                data_used.append(i_mini)
                did_cut = True
            if np.sign(pred_point_simplified_vec(i_maxi, c, X, dmat)) != y[i_maxi]:
                print(f'Cutting at iteration {it}')
                cut(Ct, X[i_maxi], y[i_maxi], dmat[i_maxi])
                data_used.append(i_maxi)
                did_cut = True
        else:
            if np.sign(pred_point_simplified_vec(i_minabs, c, X, dmat)) != y[i_minabs]:
                print(f'Cutting at iteration {it}')
                cut(Ct, X[i_minabs], y[i_minabs], dmat[i_minabs])
                data_used.append(i_minabs)
                did_cut = True
        it += 1
        
        #data_used = list(set(data_used))
        
        #print(len(data_tried))
    
    return Ct, c, data_used

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

# if __name__ == "__main__":
#     RANDOM_STATE = 0
#     X_all, y_all, X, y, X_test, y_test = generate_spiral_data(n=10, n_train=80)
#     # dmat = generate_hyperplane_arrangement(X = X, P = 2000, seed = RANDOM_STATE)
#     beta=1e-5
#     P=1000
#     n = 80
#     d = 3
#     np.random.seed(RANDOM_STATE)
#     dmat=np.empty((n,0))
#     ## Finite approximation of all possible sign patterns
#     for i in range(P):
#         u=np.random.randn(d,1)
#         dmat=np.append(dmat,drelu(np.dot(X,u)),axis=1)

#     dmat=(np.unique(dmat,axis=1))

#     C, c, used = cutting_plane(X, y, dmat, 20)
#     print(f'size of C: {len(C)}')
#     print(f'used: {used}')
#     n_train, d = X.shape
#     m = dmat.shape[1]
#     theta_matrix = np.reshape(c, (2*d, m), order='F')
#     Uopt1_final_v, Uopt2_final_v, _ = convex_solve(used, X, y, dmat)
#     Xtest = generate_Xtest(samp = 100)
#     plot_decision_boundary(X, y, X_test, y_test, Uopt1_final_v, Uopt2_final_v, used, 'Cvx')



