import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm
plt.style.use('ggplot')


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


def plot_trajectories(trajectories, target=None, ax=None, cmap=None):
    num_t = trajectories.shape[0]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if cmap is not None:
        c_ax = fig.add_axes([0.8, .25, 0.05, 0.5])
        cmap = cm.get_cmap(cmap)
        colors = cmap(np.linspace(0, 1, num_t))
        matplotlib.colorbar.ColorbarBase(ax=c_ax,
                                         ticks=range(1, num_t + 1),
                                         label='Gradient Descent Iterations',
                                         values=range(1, num_t + 1))
    alphas = np.linspace(0.25, 1, num_t, endpoint=True)
    for i, it in enumerate(trajectories):
        ax.plot(it[:, 0], it[:, 1], "k--", alpha=alphas[i])
        ax.plot(it[0, 0], it[0, 1], "or")
        if cmap is not None:
            ax.plot(it[-1, 0], it[-1, 1], "o", color=colors[i])
        else:
            ax.plot(it[-1, 0], it[-1, 1], "og")
    if target is not None:
        ax.plot(target[0], target[1], "*k", markerSize=20, label="Goal state")
    ax.legend(loc="upper left")
    ax.set_xlabel("$x$ position")
    ax.set_ylabel("$y$ position")
    return ax


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


def visualize_cost_field(S, lims=[-500, 500], target=None, ax=None):
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


def visualize_control_field(K, lims=[-500, 500], target=None, ax=None):
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
