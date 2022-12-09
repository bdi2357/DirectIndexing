import numpy as np
from qpsolvers import solve_qp

M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
P = M.T @ M  # this is a positive definite matrix
q = np.array([3.0, 2.0, 3.0]) @ M
G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
h = np.array([3.0, 2.0, -2.0])
G = np.concatenate( (G,np.identity(3)),axis=0)
h = np.array(list(h)+[1.,1.,1.])
A = np.array([1.0, 1.0, 1.0])
b = np.array([1.0])

#x = solve_qp(P, q, G, h, A, b, solver="proxqp")
#cvxopt
x = solve_qp(P, q, G, h, A, b, solver="cvxopt")

print(f"QP solution: x = {x}")