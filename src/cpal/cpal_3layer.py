"""
Implement CPAL with a three-layer ReLU Network for classification tasks.
"""

import cvxpy as cp
import numpy as np
from src.cpal.utils import predict
from src.cpal.plot import *
import warnings
warnings.filterwarnings('ignore')


def center(X, S, Di, Qj, R=1):
    _, d = X.shape
    P2 = Di.shape[1]
    P1 = Qj.shape[1]
    s = cp.Variable(4*P2*P1*d)
    obj = cp.log(R - cp.norm(s))
    constraints = []
    if len(S) > 0:
        obj += cp.sum([cp.sum(cp.log(rhs - lhs @ s)) for lhs, rhs in S])
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.MOSEK)
    return s.value

def cut(X, y, S, i, Di, Qj):
    P2 = Di.shape[1]
    P1 = Qj.shape[1]
    # inner relu constraint 
    i1 = np.kron(np.diag(2*Qj[i,:]-1),X[i,:])
    i2 = np.kron(np.eye(4*P2),i1)
    S.append((-i2,0))
    # outter relu constraint 
    o1 = np.kron(Qj[i,:],X[i,:])
    o2 = np.kron(np.diag(2*Di[i,:]-1), o1)
    o3 = np.kron(np.array([1,-1]),o2)
    o4 = np.kron(np.eye(2),o3)
    S.append((-o4,0))
    # prediction constraint 
    p1 = np.kron(Qj[i,:],X[i,:])
    p2 = np.kron(np.diag(Di[i,:]), p1)
    p3 = np.kron(np.array([1,-1,-1,1]),p2)
    S.append((-y[i]*(p3.sum(axis=0)),0))

def query(X, c, Di, Qj, data_used, data_tried):
    n_train, _ = X.shape
    mini = np.inf
    i_mini = -1
    maxi = -np.inf
    i_maxi = -1
    minabs = np.inf
    i_minabs = -1
    for i in range(n_train): 
        if (i not in data_used) and (i not in data_tried):
            pred = predict(X, c, i, Di, Qj)
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


def cutting_plane_3layer(X, y, Di, Qj, n_points):
    n_train, _ = X.shape
    data_used = []
    data_tried = []
    plane = []
    it = 0
    while len(data_used) < n_train and it < n_points:
        c = center(X, plane, Di, Qj)
        i_mini, i_maxi, _ = query(X, c, Di, Qj, data_used, data_tried)
        data_tried += [i_mini, i_maxi]
        if np.sign(predict(X, c, i_mini, Di, Qj)) != y[i_mini]:
            cut(X, y, plane,i_mini, Di, Qj)
            data_used.append(i_mini)
        if np.sign(predict(X, c, i_maxi, Di, Qj)) != y[i_maxi]:
            cut(X,y, plane, i_maxi, Di, Qj)
            data_used.append(i_maxi)
        it += 1     
        print('cutting plane iteration: ', it, ' # data used: ', data_used)    
    return c, data_used