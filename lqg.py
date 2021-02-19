import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from loguru import logger
from utils import plot
"""

TODO
- TIDY this up, into package and scripts
- prove that K is invariant to goal
    - just feed back the task error to reach a set point!
        - the cost want's the state, whatever it is, to go to zero.
    - K, S, etc is translation invariant to the goal state
    - to build this translation into the state/dynamics, we augment the state/cost with the target
        - this just computes the error from that state internally
    - write about this:
        - L is linear in state, we can apply linear operations to it?
        - S is quadratic in state, but we can also linearly transform it?
- multiple targets?
    - this doesn't work by adding another target, doesn't converge
    - would need to decide between targets?
    - are value functions additive?
- make time, force, etc into reasonable units

Goals
- finish cleaning this up
    - X infinite horizon lqr
    - "" "" with multiplicative noise
    - model corruption
    - model adaptation gradient
        - compute gradient? diagonalizable?
    - write up discussion
        - composing LQR policies
            - averaging
            - in what case is composition useful?

- analyze andy's data
    - draw specific plots i want
    - behavioral trajectories
    - EMG features -- low-dimensional trajectories
    - comparisons within sessions
    - comparisons across sessions

"""

fmt = "{time} - {name} - {level} - {message}"
logger.add("debug.log", level="DEBUG", format=fmt)
logger.add(sys.stderr, level="ERROR", format=fmt)


class LQR():
    def __init__(self, dynamic_params, control_params):
        self.target = cost_params['target']
        self.dynamics_params = dynamic_params
        self.cost_params = control_params
        self.A, self.B = self.compute_dynamics()
        self.R, self.Q = self.compute_costs()
        self.S = self.compute_steady_state_cost_to_go()
        self.K = self.compute_control_law()

    def compute_dynamics(self):
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
        dt = self.dynamics_params["dt"]
        mass = self.dynamics_params["mass"]
        force_time_constant = self.dynamics_params["force_time_constant"]
        state_noise_magnitud = self.dynamics_params["state_noise_magnitude"]
        state_dim = self.dynamics_params["state_dim"]
        state_noise_covariance = self.dynamics_params["state_noise_covariance"]

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
        | 1  0  dt 0  0    0   
        | 0  1  0  dt 0    0   
        | 0  0  1  0  dt/m 0   
        | 0  0  0  1  0   dt/m 
        | 0  0  0  0  e^-x 0   
        | 0  0  0  0  0   e^-x 
        """

        # control acts on force only
        B = np.eye(state_dim)
        B = B[:, 4:6]

        return A, B

    def compute_costs(self):

        dt = cost_params["dt"]
        target = self.target
        velocity_cost_amplitude = cost_params["velocity_cost_amplitude"]
        force_cost_amplitude = cost_params["force_cost_amplitude"]
        control_cost_magnitude = cost_params["control_cost_magnitude"]

        T = 1
        times = np.linspace(0, T, int(T / dt))

        # state cost x^TQx = (x^TD^T)(Dx)
        D_target = np.array([[-1, 0, 0, 0, 0, 0, target[0], 0],\
                             [ 0,-1, 0, 0, 0, 0, 0, target[1]]])
        # [0,  0, velocity_cost_amplitude, 0,  0,  0, 0,0],\
        # [0,  0, 0,  velocity_cost_amplitude, 0,  0, 0,0],\
        # [0,  0, 0,  0,  force_cost_amplitude, 0, 0  ,0],\
        # [0,  0, 0,  0,  0,  force_cost_amplitude, 0,0]])

        Q = D_target.T @ D_target

        # control cost u^TRu -- just a plain magnitude cost
        R = control_cost_magnitude * np.eye(2)

        return R, Q

    def compute_steady_state_cost_to_go(self):
        A = self.A
        B = self.B
        R = self.R
        Q = self.Q
        i = 0
        tol = 0.00001
        old_S = Q
        while True:
            i += 1
            if i > 5000:
                logger.error(
                    "Exceeded maximum number of steady state computation iterations."
                )
                break
            S = self.backward_recurse(old_S)
            if np.max(np.abs(old_S - S)) < tol:
                logger.info("Converged in {i} steps.", i=i)
                return S
            old_S = S

    def backward_recurse(self, S):
        A = self.A
        B = self.B
        R = self.R
        Q = self.Q
        return A.T @ S @ A - (A.T @ S @ B) @ np.linalg.inv(R + B.T @ S @ B) @ (
            B.T @ S @ A) + Q

    def compute_control_law(self):
        A = self.A
        B = self.B
        R = self.R
        S = self.S
        return -np.linalg.inv(R + B.T @ S @ B) @ (B.T @ S @ A)

    def advance_dynamic(self, x, state_noise):
        u = self.K @ x
        x = self.A @ x + self.B @ u + state_noise
        return x, u

    def sample_state_noise(self, state_dim):
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
        state_noise_magnitude = self.dynamics_params["state_noise_magnitude"]
        state_noise_covariance = self.dynamics_params["state_noise_covariance"]
        return state_noise_magnitude * np.random.default_rng(
        ).multivariate_normal(np.zeros(state_dim),
                              state_noise_covariance).reshape(-1, 1)

    def sample_trajectory(self, x0, steps, perturbation=None):
        A = self.A
        B = self.B
        K = self.K
        state_noise_magnitude = self.dynamics_params["state_noise_magnitude"]
        state_noise_covariance = self.dynamics_params["state_noise_covariance"]
        x_list = [x0]
        u_list = []
        for i in range(steps):
            state_noise = self.sample_state_noise(x_list[i].shape[0])
            if perturbation is not None:
                x_in = np.add(perturbation[:, i].reshape(-1, 1), x_list[i])
            else:
                x_in = x_list[i]
            x, u = self.advance_dynamic(x_in, state_noise)
            x_list.append(x.reshape(-1, 1))
            if i != steps - 1:
                u_list.append(u.reshape(-1, 1))
        return np.array(x_list)[:, :, 0], np.array(u_list)[:, :, 0]

    def backward_pass(A, B, R, Q):
        """
            For finite time control and cost-to-go list
        """
        S = Q
        S_list = [S]
        L_list = []
        L = compute_control_law(A, B, R, Q, S)
        S = backward_recurse(A, B, R, Q, S)
        S_list.append(S)
        L_list.append(L)
        L_list = L_list[::-1]
        S_list = S_list[::-1]
        return np.array(L_list), np.array(S_list)


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


def visualize_cost_field(S, lims=[-500, 500], ax=None):
    """
    plot the cost over the state space for a given LQR problem

    """
    def f(x, S):
        return x.T @ S @ x

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    n = lims[1] - lims[0]
    J = np.zeros(shape=(n, n))
    for i, x in enumerate(np.linspace(lims[0], lims[1], n)):
        for j, y in enumerate(np.linspace(lims[0], lims[1], n)):
            state = np.array([x, y, 0, 0, 0, 0, 1, 1]).reshape(-1, 1)
            J[i][j] = f(state, S)
    im = ax.imshow(J.T,
                   origin="lower",
                   extent=[lims[0], lims[1], lims[0], lims[1]])
    if target is not None:
        ax.plot(target[0], target[1], "*w", markerSize=20)
    fig.set_frameon(False)
    fig.colorbar(im)
    return ax


def visualize_control_field(K, lims=[-500, 500], ax=None):
    """
    plot the cost over the state space for a given LQR problem

    """
    def f(x, K):
        return K @ x

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    n = (lims[1] - lims[0]) // 50
    u = np.zeros(shape=(n, n, 2))
    xx = np.linspace(lims[0], lims[1], n)
    yy = np.linspace(lims[0], lims[1], n)
    xg, yg = np.meshgrid(xx, yy)
    for i, x in enumerate(xx):
        for j, y in enumerate(yy):
            state = np.array([x, y, 0, 0, 0, 0, 1, 1]).reshape(-1, 1)
            u[i][j] = f(state, K)[:2].reshape(-1)
    ax.quiver(yy, xx, u[:, :, 0].T, u[:, :, 1].T)
    if target is not None:
        ax.plot(target[0], target[1], "*w", markerSize=20)
    return ax


def plot_eigenvalues(matrix, type):
    fig, ax = plt.subplots(1, 1)
    vals, vecs = np.linalg.eig(matrix)
    poles = np.vstack([np.real(vals), np.imag(vals)]).T
    if type == "arrow":
        r = np.max(poles) + 0.2
        ax.set_xlim([-r, r])
        ax.set_ylim([-r, r])
        cc = plt.Circle((0.0, 0.0), 1.0, alpha=0.1)
        ax.set_aspect(1)
        ax.add_artist(cc)
        for p in poles:
            ax.arrow(0,
                     0,
                     p[0],
                     p[1],
                     color='k',
                     width=0.01,
                     head_length=0.075,
                     head_width=0.1)
    elif type == "bar":
        ax.bar(range(1, poles.shape[0] + 1),
               poles[:, 0],
               color="g",
               label="real")
        ax.bar(range(1, poles.shape[0] + 1),
               poles[:, 1],
               color="y",
               label="imaginary")
        plt.plot([1, poles.shape[0]], [1, 1], "k--")
        ax.legend(loc="lower center")


def sample_trajectories(lqr, num_steps, lims=[-500, 500], perturbation=None):
    trajectories = []
    for i in range(num_movements):
        x0 = np.array([
            np.random.uniform(lims[0] // 2, lims[1] // 2),
            np.random.uniform(lims[0] // 2, lims[1] // 2), 0, 0, 0, 0, 1, 1
        ]).reshape(-1, 1)
        states, controls = lqr.sample_trajectory(x0,
                                                 num_steps,
                                                 perturbation=perturbation)
        trajectories.append(states)
    return np.array(trajectories)


if __name__ == '__main__':
    target = [-100, -100, 100, 100]
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


    def plot_trajectories(trajectories, target=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for it in trajectories:
            ax.plot(it[:, 0], it[:, 1], "k--", alpha=0.5)
            ax.plot(it[0, 0], it[0, 1], "og")
            ax.plot(it[-1, 0], it[-1, 1], "or")
        if target is not None:
            ax.plot(target[0], target[1], "*w", markerSize=5)
        return ax

    ax = visualize_cost_field(lqr1.S)
    ax = visualize_control_field(lqr1.K)
    # plot_trajectories(trajectories_plain, target=target, ax=ax)
    # ax = visualize_cost_field(lqr.S, lims)
    # plot_trajectories(trajectories_perturbed, target=target, ax=ax)
    # ax = visualize_cost_field(lqr.S, lims)
    # plot_trajectories(trajectories_noise, target=target, ax=ax)
    plt.show()