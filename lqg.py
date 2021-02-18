import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from loguru import logger
from utils import plot
"""

TODO
- set up finite time nicely
    - what are the constraints here?
    - what checks are necessary?
- do infinite horizon with truncation
    - this should work online?
    - is this goal-specific?
- make this work online?
    - what cases can respond to perturbations?
- how to inferface with e.g. mujoco?

"""
"""

questions
- what does time look like here?

"""
"""

types
- finite time, single goal
- finite time, trajectory
- infinite time, single goal

"""

fmt = "{time} - {name} - {level} - {message}"
logger.add("debug.log", level="DEBUG", format=fmt)
logger.add(sys.stderr, level="ERROR", format=fmt)


def sample_state_noise(state_dim, state_noise_magnitude,
                       state_noise_covariance):
    """
    Sample a Gaussian state noise vector
    Args:
        x: state needed for the state_dimension
        noise_mag: amplitude of the gaussian
        Omega: covariance

    Returns:
        Sample from a centered, multivariate Gaussian.

    Raises:
        
    Notes:
        We could handle the random seed more carefully here?

    """
    return state_noise_magnitude * np.random.default_rng().multivariate_normal(
        np.zeros(state_dim), state_noise_covariance).reshape(-1, 1)


def compute_dynamics(dt, mass, force_time_constant, state_noise_magnitude,
                     state_dim, state_noise_covariance):
    """
    Args:
        dt: 
        target: 
        tau: 
        m: 
        r: 
        noise_mag: 
        wv: 
        wf: 
   
    Returns:
        This is a description of what is returned.

    Raises:
        

    """

    T = 1
    times = np.linspace(0, T, int(T / dt))

    # dynamics
    A = np.eye(state_dim)
    A[0][2] = dt
    A[1][3] = dt
    A[2][4] = dt / mass
    A[3][5] = dt / mass
    A[4][4] = np.exp(-dt / force_time_constant)
    A[5][5] = np.exp(-dt / force_time_constant)
    """
    A: 
    | 1  0  dt 0  0    0   0
    | 0  1  0  dt 0    0   0
    | 0  0  1  0  dt/m 0   0
    | 0  0  0  1  0   dt/m 0
    | 0  0  0  0  e^-x 0   0
    | 0  0  0  0  0   e^-x 0
    | 0  0  0  0  0   0    1
    """

    # control acts on force only
    B = np.array(\
        [[0,0],\
         [0,0],\
         [0,0],\
         [0,0],\
         [1,0],\
         [0,1],\
         [0,0]])

    return A, B


def compute_costs(dt, target, velocity_cost_amplitude, force_cost_amplitude,
                  control_cost_magnitude):

    T = 1
    times = np.linspace(0, T, int(T / dt))

    # state cost x^TQx = (x^TD^T)(Dx)
    D_target = np.array([
        [-1, 0, 0,  0,  0,  0, target[0]],\
        [0, -1, 0,  0,  0,  0, target[1]],\
        [0,  0, velocity_cost_amplitude, 0,  0,  0,  0],\
        [0,  0, 0,  velocity_cost_amplitude, 0,  0,  0],\
        [0,  0, 0,  0,  force_cost_amplitude, 0,  0],\
        [0,  0, 0,  0,  0,  force_cost_amplitude, 0]])

    Q_list = []
    for i, t in enumerate(times):
        Q_list.append((1 / (len(D_target) + 1)) * (D_target.T @ D_target))

    # control cost u^TRu -- just a plain magnitude cost
    R = control_cost_magnitude * np.eye(2) / len(times)

    return R, Q_list


def backward_recurse(A, B, R, Q, S):
    return A.T @ S @ A - (A.T @ S @ B) @ np.linalg.inv(R + B.T @ S @ B) @ (
        B.T @ S @ A) + Q


def compute_steady_state_cost_to_go(A, B, R, Q, S):
    i = 0
    tol = 0.00001
    while True:
        i += 1
        if i > 1000:
            logger.error("Exceeded steady stay computation recursion.")
            break
        old_S = S
        S = backward_recurse(A, B, R, Q, old_S)
        if np.max(np.abs(old_S - S)) < tol:
            logger.info("Converged in {i} steps.", i=i)
            return S


def plot_eigenvalues(matrix, color, figax=None):
    if figax is None:
        figax = plt.subplots(1, 1)
    vals, _ = np.linalg.eig(matrix)
    figax[1].plot(np.real(vals), "o" + color)
    return figax


def compute_control_law(A, B, R, Q, S):
    return -np.linalg.inv(R + B.T @ S @ B) @ (B.T @ S @ A)


def backward_pass(A, B, R, Q_list):
    S = Q_list[-1]
    S_list = [S]
    L_list = []
    for Q in Q_list[::-1]:
        L = compute_control_law(A, B, R, Q, S)
        S = backward_recurse(A, B, R, Q, S)
        S_list.append(S)
        L_list.append(L)
    L_list = L_list[::-1]
    S_list = S_list[::-1]
    return np.array(L_list), np.array(S_list)


def advance_dynamic(x, A, B, K, state_noise):
    u = K @ x
    x = A @ x + B @ u + state_noise
    return x, u


def forward_pass(x0, A, B, L_list, state_noise_magnitude,
                 state_noise_covariance):
    x_list = [x0]
    u_list = []
    num_iterations = len(L_list)
    for i in range(num_iterations):
        state_noise = sample_state_noise(x_list[i].shape[0],
                                         state_noise_magnitude,
                                         state_noise_covariance)
        x, u = advance_dynamic(x_list[i], A, B, L_list[i], state_noise)
        x_list.append(x.reshape(-1, 1))
        if i != num_iterations - 1:
            u_list.append(u.reshape(-1, 1))
    return np.array(x_list)[:, :, 0], np.array(u_list)[:, :, 0]


if __name__ == '__main__':
    x0 = np.array([0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1)
    state_dim = 7
    dynamics_params = {
        "state_dim": state_dim,
        "dt": 0.005,
        "force_time_constant": 0.040,  # 40ms, exponential filter constant
        "mass": 1,
        "state_noise_magnitude": 0.01,
        "state_noise_covariance": np.eye(state_dim)
    }

    cost_params = {
        "dt": 0.005,
        "control_cost_magnitude": 0.01,
        "velocity_cost_amplitude": 0.01,
        "force_cost_amplitude": 0.01,
        "target": np.array([50, 50])
    }

    num_movements = 10

    def plot_trajectory_pair(trajectories, ylabels, goal=None):
        fig, axes = plt.subplots(len(ylabels), 1, figsize=(10, 10))
        for i, ax in enumerate(axes):
            for trajectory in trajectories:
                ax.plot(trajectory[:, i])
                ax.set_xlabel("Time")
                ax.set_ylabel(ylabels[i])
            if goal is not None:
                ax.plot(trajectory.shape[0], goal[i], "or")
        return fig, axes

    def compute_state_cost_to_go(S):
        def f(x, S):
            return x.T @ S[:2, :2] @ x

        xaxis = np.linspace(0, 1, 100)
        yaxis = np.linspace(0, 1, 100)
        x = np.vstack([xaxis, yaxis])
        return f(x, S)

    def run_infinite_time():
        A, B = compute_dynamics(**dynamics_params)
        R, Q_list = compute_costs(**cost_params)
        S = compute_steady_state_cost_to_go(A, B, R, Q_list[-1], Q_list[-1])
        K = compute_control_law(A, B, R, Q_list[50], S)
        trajectories = []
        for i in range(num_movements):
            states, controls = forward_pass(
                x0, A, B, [K for _ in range(200)],
                dynamics_params["state_noise_magnitude"],
                dynamics_params["state_noise_covariance"])
            trajectories.append(states)
        trajectories = np.array(trajectories)

        # fig, axes = plot_trajectory_pair(trajectories[:, :, :2],
        #                                  ylabels=["x position", "y position"],
        #                                  goal=cost_params["target"])
        # fig, axes = plot_trajectory_pair(trajectories[:, :, 4:6],
        #                                  ylabels=["x force", "y force"])
        return K, S, trajectories

    def run_finite_time():
        A, B = compute_dynamics(**dynamics_params)
        R, Q_list = compute_costs(**cost_params)
        trajectories = []
        for i in range(num_movements):
            L_list, S_list = backward_pass(A, B, R, Q_list)
            states, controls = forward_pass(
                x0, A, B, L_list, dynamics_params["state_noise_magnitude"],
                dynamics_params["state_noise_covariance"])
            trajectories.append(states)
        trajectories = np.array(trajectories)

        # fig, axes = plot_trajectory_pair(trajectories[:, :, :2],
        #                                  ylabels=["x position", "y position"],
        #                                  goal=cost_params["target"])
        # fig, axes = plot_trajectory_pair(trajectories[:, :, 4:6],
        #                                  ylabels=["x force", "y force"])
        return L_list, S_list, trajectories

    K, S, infinite_trajectories = run_infinite_time()

    def visualize_cost_field(S):
        """
        plot the cost over the state space for a given LQR problem

        """
        fig, ax = plt.subplots(1, 1)
        J = compute_state_cost_to_go(S)
        im = ax.imshow(J, origin="lower")
        fig.set_frameon(False)
        fig.colorbar(im)
        return fig, ax

    def visualize_control_law(K):
        """
        plot the cost over the state space for a given LQR problem

        """
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(K, origin="upper")
        # Major ticks
        ax.set_xticks(np.arange(0, K.shape[1], 1))
        ax.set_yticks(np.arange(0, K.shape[0], 1))
        # Labels for major ticks
        ax.set_xticklabels(np.arange(1, K.shape[1] + 1, 1))
        ax.set_yticklabels(np.arange(1, K.shape[0] + 1, 1))
        # Minor ticks
        ax.set_xticks(np.arange(-.5, K.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, K.shape[0], 1), minor=True)
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        fig.set_frameon(False)
        fig.colorbar(im)
        return fig, ax

    # figax = plot_eigenvalues(K.T @ K, color="k")
    # fig, ax = plt.subplots(1, 1)
    # for L in L_list:
    #     plot_eigenvalues(L.T @ L, color="b", figax=(fig, ax))

    figax = visualize_cost_field(S)
    figax[1].set_xlim([0, 100])
    figax[1].set_ylim([0, 100])
    for it in infinite_trajectories:
        figax[1].plot(it[:, 0], it[:, 1])

    # figax = visualize_control_law(K)

    plt.show()