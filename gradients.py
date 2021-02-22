from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.linalg import matrix_power

# https://mathoverflow.net/questions/384464/gradient-descent-for-markov-dynamics


def check_inputs(A, B, K, n, v, w):
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2
    A_rows = dim[0]
    A_cols = dim[1]
    assert isinstance(B, np.ndarray)
    dim = B.shape
    assert len(dim) == 2
    B_rows = dim[0]
    B_cols = dim[1]
    assert isinstance(K, np.ndarray)
    dim = K.shape
    assert len(dim) == 2
    K_rows = dim[0]
    K_cols = dim[1]
    if isinstance(n, np.ndarray):
        dim = n.shape
        assert dim == (1, )
    assert isinstance(v, np.ndarray)
    dim = v.shape
    assert len(dim) == 1
    x_rows = dim[0]
    assert isinstance(w, np.ndarray)
    dim = w.shape
    assert len(dim) == 1
    y_rows = dim[0]
    assert B_rows == y_rows == A_rows
    assert K_cols == A_cols == x_rows
    assert B_cols == K_rows


def elementwise(A, B, K, n, v, w):
    check_inputs(A, B, K, n, v, w)
    T_0 = A - (B @ K)
    t_1 = (T_0**n).dot(v) - w
    f = np.linalg.norm(t_1)**2
    # g = ((2 * n) * ((t_1[:, np.newaxis] *
    #                         (T_0**(n - 1))) * v[np.newaxis, :]))
    g = (2 * n) * np.diag(t_1) @ (T_0**(n - 1)) @ np.diag(v)
    return f, g


def check_elementwise_gradient(A, B, K, n, v, w):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(*A.shape)
    f1, _ = elementwise(A + t * delta, B, K, n, v, w)
    f2, _ = elementwise(A - t * delta, B, K, n, v, w)
    f, g = elementwise(A, B, K, n, v, w)
    print('elementwise approximation error',
          np.linalg.norm((f1 - f2) / (2 * t) - np.tensordot(g, delta, axes=2)))


# If you have any advice or references for computing these derivatives, I would be grateful!  Thank you so much for your time. I've confirmed numerically your solutions, and have discovered that the online calculator is performing elementwise exponentiation, which should explain the discrepancy. For context, these derivatives crop up as I'm working to learn a Markov transition function via gradient descent. I'd love to continue to second-order, if you have any suggestions!


def scalar(A, B, K, n, v, w):
    # 2𝑁(𝑀^𝑁𝑣 − 𝑤)^𝑇𝑀^(𝑁−1)𝑣
    check_inputs(A, B, K, n, v, w)
    M = A - (B @ K)
    Mn = matrix_power(M, n)
    Mnm1 = matrix_power(M, n - 1)
    f = np.linalg.norm(Mn @ v[:, None] - w[:, None])**2
    g = (2 * n *
         (Mn @ v[:, None] - w[:, None]).T @ Mnm1 @ v[:, None]).flatten()[0]
    return f, g


def check_scalar_gradient(A, B, K, n, v, w):
    t = 1E-6
    # f(A + eps*I) = eps*(df/dA)
    delta = np.eye(A.shape[0]) * t
    f1, _ = scalar(A + delta, B, K, n, v, w)
    f2, _ = scalar(A - delta, B, K, n, v, w)
    gn = (f1 - f2) / (2 * t)
    f, g = scalar(A, B, K, n, v, w)
    print('scalar approximation error', ((f1 - f2) / (2 * t)) - g)


def full(A, B, K, n, v, w):
    g = np.zeros(A.shape)
    M = A - (B @ K)
    Mn = matrix_power(M, n)
    f = np.linalg.norm((Mn @ v[:, None]) - w[:, None])**2
    for k in range(1, n + 1):
        x = matrix_power(M, k - 1).T @ ((Mn @ v[:, None]) - w[:, None])
        y = matrix_power(M, n - k) @ v[:, None]
        g = g + 2 * np.outer(x, y)
    return f, g


def check_full_gradient(A, B, K, n, v, w):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(*A.shape)
    f1, _ = full(A + t * delta, B, K, n, v, w)
    f2, _ = full(A - t * delta, B, K, n, v, w)
    f, g = full(A, B, K, n, v, w)
    ng = (f1 - f2) / (2 * t)
    print('full approximation error',
          (f1 - f2) / (2 * t) - np.tensordot(g, delta))


def generateRandomData():
    A = np.random.randn(3, 3)
    B = np.zeros((3, 3))
    K = np.zeros((3, 3))
    n = np.random.randint(low=2, high=10)
    v = np.random.randn(3)
    w = np.ones((3))
    return A, B, K, n, v, w


# if __name__ == '__main__':
# A, B, K, n, v, w = generateRandomData()
# check_elementwise_gradient(A, B, K, n, v, w)
# check_scalar_gradient(A, B, K, n, v, w)
# check_full_gradient(A, B, K, n, v, w)
# is the scalar derivative the trace of the full?
# f, g = scalar(A, B, K, n, v, w)
# f, g = full(A, B, K, n, v, w)
