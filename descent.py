from viz import plot_trajectories, visualize_cost_field
import numpy as np
from lqr import LQR, sample_trajectories
from gradients import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def is_pos_def(X):
    return np.all(np.linalg.eigvals(X) > 0)


def add_submatrix(S, A):
    s = A.shape
    Sc = np.copy(S)
    Sc[:s[0], :s[1]] += A
    return Sc


def diagonalize(X):
    if is_pos_def(X):
        u, V = np.linalg.eig(X)
        return V, np.diag(u), np.linalg.inv(V)
    else:
        raise ValueError("Matrix not positive definite.")


def mat_power(M, power):
    P, D, Pinv = diagonalize(M)
    return P @ (D**power) @ Pinv


def remove_diagonal(A):
    Ac = A.copy()
    return Ac - np.diag(Ac.diagonal())


def add_diagonal_noise_to_A(A, A_noise_mag):
    """
        Adds gausian noise to the diagonal
    """
    A = np.copy(A)
    state_dim = A.shape[0] - 2
    noise = A_noise_mag * np.diag(np.random.randn(state_dim))
    A = add_submatrix(A, noise)
    return A


def add_friction_to_A(A, friction):
    A = np.copy(A)
    state_dim = A.shape[0] - 2
    A = add_submatrix(A, -np.eye(state_dim - 2) * friction)
    return A


def print_eigenvalues(M):
    print(np.linalg.eigvals(M[:state_dim, :state_dim]))


target = [0, 1]
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
    "velocity_cost_amplitude": .05,
    "force_cost_amplitude": 0.01,
    "target": np.array(target)
}

x0 = np.zeros((state_dim + 2))
x0[-2:] = np.array([1, 1])
x0[:2] = np.array([.001, .001])
t = np.array([target[0], target[1], 0, 0, 0, 0, 1, 1])

lqr = LQR(dynamics_params, cost_params)
original_A = np.copy(lqr.A)

noise_mag = 0.000001
friction_mag = 0.0
lqr.A = add_friction_to_A(lqr.A, friction_mag)
lqr.A = add_diagonal_noise_to_A(original_A, noise_mag)
lqr.reoptimize()
unoptimized_A = np.copy(lqr.A)

num_steps = 200
num_movements = 20
learning_rate = 1E-8
noisy_A_trajectories = np.empty((num_movements, num_steps + 1, state_dim + 2))
for i in range(num_movements):
    x, u = lqr.sample_trajectory(x0.reshape(-1, 1), num_steps)
    noisy_A_trajectories[i, :, :] = x
    # gradient of the error e
    e, grad = full(lqr.A[:-2, :-2], lqr.B[:-2, :], lqr.K[:, :-2], num_steps,
                   x0[:-2], t[:-2])
    # print("error:", e)
    lqr.A = add_submatrix(lqr.A, -(learning_rate * grad))
    # lqr.reoptimize()

plot_trajectories(noisy_A_trajectories, target=target, cmap="viridis")
plt.show()