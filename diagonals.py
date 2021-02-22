from viz import plot_trajectories, visualize_cost_field
import numpy as np
from lqr import LQR, sample_trajectories


def is_pos_def(X):
    return np.all(np.linalg.eigvals(X) > 0)


def diagonalize(X):
    if is_pos_def(X):
        u, V = np.linalg.eig(X)
        return V, np.diag(u), np.linalg.inv(V)
    else:
        raise ValueError("Matrix not positive definite.")


def mat_power(M, power):
    P, D, Pinv = diagonalize(M)
    return P @ (D**power) @ Pinv


target = [100, 100]
state_dim = 6
state_noise_covariance = np.eye(state_dim)
dynamics_params = {
    "state_dim": state_dim,
    "dt": 0.005,
    "force_time_constant": 0.040,  # 40ms, exponential filter constant
    "mass": 1,
    "state_noise_magnitude": .0001,
    "state_noise_covariance": state_noise_covariance
}
cost_params = {
    "dt": 0.005,
    "control_cost_magnitude": 1,
    "velocity_cost_amplitude": 0.05,
    "force_cost_amplitude": 0.05,
    "target": np.array(target)
}

x0 = np.random.randn(state_dim + 2, 1)
x0[-2:] = np.array([1, 1]).reshape(-1, 1)

t = np.array([target[0], target[1], 0, 0, 0, 0, 1, 1]).reshape(-1, 1)

lqr = LQR(dynamics_params, cost_params)
# noisy A matrix
noise = np.zeros((state_dim + 2, state_dim + 2))
noise[:state_dim, :state_dim] = np.random.uniform(low=-0.01,
                                                  high=0,
                                                  size=(state_dim, state_dim))
noise -= np.diag(noise.diagonal())
lqr.A += noise
d = lqr.A.diagonal()
lqr.A -= np.diag(d)
lqr.A += np.diag(0.9 * np.ones((state_dim + 2)))
print(lqr.A)
nu = 0.0001
trajectories = []
for i in range(10):
    trajectory = lqr.sample_trajectory(x0, 200)
    grad = ddA(lqr.A[:-2, :-2], lqr.B[:-2, :], lqr.K[:, :-2], x0[:-2, 0],
               t[:-2, 0], 20)
    print("grad: ", grad)
    input()
    g = np.zeros((state_dim + 2, state_dim + 2))
    g[:state_dim, :state_dim] += grad
    lqr.A = np.add(lqr.A, nu * g)
    trajectories.append(trajectories)
    print(lqr.A)
trajectories = np.array(trajectories)
ax = visualize_cost_field(lqr.S)
plot_trajectories(trajectories, ax=ax)

# compare numerical and analytic derivative
# k = 5
# A = np.eye(k)
# B = np.zeros((k, k // 2))
# K = np.zeros((k // 2, k))
# v = np.zeros((k, 1)) + 1
# w = np.zeros((k, 1))
# N = 10
# print(ddA(A, B, K, v, w, N))
# print(ddA_numeric(A, B, K, v, w, N))