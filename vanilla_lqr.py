from lqr import LQR, sample_trajectories
from viz import *
import numpy as np
import matplotlib.pyplot as plt

target = [100, 100]
state_dim = 8
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

num_movements = 10
num_steps = 250
state_space = 1000

perturbation = np.zeros((state_dim, num_steps))
# perturbation[4, 10:50] = 500

lqr1 = LQR(dynamics_params, cost_params)

# lqr = LQR(dynamics_params, cost_params)
# lqr.A[:-2, :-2] += 0.001 * np.random.normal(size=lqr.A[:-2, :-2].shape)
# trajectories_noise = sample_trajectories(lqr,
#                                          num_steps,
#                                          perturbation=perturbation)

# The closed-loop state transfer matrix is stable
# if and only if all of its eigenvalues are strictly
# inside the unit circle of the complex plane.

# figax = plot_eigenvalues((lqr.A - (lqr.B @ lqr.K)), "bar")
# figax = plot_eigenvalues(lqr.A, "bar")

# figax = visualize_control_law(lqr.S)
# figax = visualize_control_law(lqr.K)

# fig, axes = plot_trajectory_pair(trajectories[:, :, :2],
#                                  ylabels=["x position", "y position"],
#                                  goal=cost_params["target"])
# fig, axes = plot_trajectory_pair(trajectories[:, :, 4:6],
#                                  ylabels=["x force", "y force"])

# figax = visualize_cost_field(lqr.S)

# fig, ax = visualize_control_field(lqr.K, lims)

ax = visualize_cost_field(lqr1.S)
ax = visualize_control_field(lqr1.K)
# plot_trajectories(trajectories_plain, target=target, ax=ax)
# ax = visualize_cost_field(lqr.S, lims)
# plot_trajectories(trajectories_perturbed, target=target, ax=ax)
# ax = visualize_cost_field(lqr.S, lims)
# plot_trajectories(trajectories_noise, target=target, ax=ax)

plt.show()