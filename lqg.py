import numpy as np
from matplotlib import pyplot as plt

p = np.random.normal([-250, 0], [10, 10])
px0 = p[0]
py0 = p[1]
print(p)
'''
Required data to set up the task
- number of time points, discretization in time
- cost at each time step
-

'''


def polar_to_cartesian(coord):
    return [coord[0] * np.cos(coord[1]), coord[0] * np.sin(coord[1])]


def cartesian_to_polar(coord):
    output = [0, 0]
    output[0] = np.sqrt(coord[0]**2 + coord[1]**2)
    if coord[1] >= 0:
        output[1] = np.arccos(coord[0] / output[0])
    else:
        output[1] = -np.arccos(coord[0] / output[0])
    return output


def sample_state_noise(x, noise_mag, Omega):
    return noise_mag * np.random.multivariate_normal(np.zeros(x.shape[0]),
                                                     Omega).reshape(-1, 1)


def generate_APT_targets(dt):
    T = 1  # sec
    times = np.linspace(0, T, int(T / dt))
    # target point for every timestep
    # APT task -- polar coordinates to cartesian
    # radius and angle, [r, theta]
    radius = 250
    angles = np.linspace(np.pi, 0, len(times), endpoint=True)
    radii = np.ones(len(angles)) * radius
    polar_coords = zip(radii, angles)
    cartesian_coords = [polar_to_cartesian(p) for p in polar_coords]
    # target: (x,y,timestep)
    return [(c[0], c[1], t)
            for c, t in zip(cartesian_coords, range(len(times)))]


def visualize_cost_field():
    pass


def generate_dynamics(dt, targets, tau, m, r, noise_mag, wv, wf):
    T = 1
    times = np.linspace(0, T, int(T / dt))

    #     endpoint = targets[-1][:-1] # x and y coords
    #     print("start: ",targets[0][:-1])
    #     print("end: ",targets[-1][:-1])

    # initial conditions
    px0 = targets[0][0]
    py0 = targets[0][1]
    # random start
    # p = np.random.normal([targets[0][0],targets[0][1]], [10,10])
    # if p[1] < 0:
    # p[1] = -p[1]
    vx0 = 0.0
    vy0 = 0.0
    fx0 = -10.0
    fy0 = 0.0
    x = np.array([px0, py0, vx0, vy0, fx0, fy0, 1]).reshape(-1, 1)

    # dynamics
    A = np.eye(x.shape[0])
    A[0][2] = dt
    A[1][3] = dt
    A[2][4] = dt / m
    A[3][5] = dt / m
    A[4][4] = np.exp(-dt / tau)
    A[5][5] = np.exp(-dt / tau)

    # dynamics noise
    Omega = np.eye(x.shape[0])

    ## task error
    # via points --> px, py - tx, ty
    #     D_targets = [np.array([
    #                 [-1,  0, 0, 0, 0, 0, target[0]],\
    #                 [0, -1, 0, 0, 0, 0, target[1]]])\
    #               for target in targets\
    #              ]
    #     # end point
    # zero velocity
    wvs = np.append(np.zeros(len(targets) // 2),
                    np.linspace(0, wv,
                                len(targets) // 2))
    wfs = np.append(np.zeros(len(targets) // 2),
                    np.linspace(0, wv,
                                len(targets) // 2))
    D_targets = [np.array([
        [-1, 0, 0,  0,  0,  0,  target[0]],\
        [0, -1, 0,  0,  0,  0,  target[1]],\
        [0,  0, wv, 0,  0,  0,  0],\
        [0,  0, 0,  wv, 0,  0,  0],\
        [0,  0, 0,  0,  wf, 0,  0],\
        [0,  0, 0,  0,  0,  wf, 0]])\
        for wv,wf,target in zip(wvs,wfs,targets)\
    ]
    Q_list = []
    ttimes = [t[2] for t in targets]
    #     print("num targets: ",len(ttimes))
    #     print("num timesteps: ",len(times))
    for i, t in enumerate(times):
        # add constraints to list of zero matrices
        #         if i == len(times)-1:
        #             print(np.max(D_end))
        #             Q_list.append((1/(len(D_targets)+1))*(D_end.T@D_end))
        if i in ttimes:
            # (add targets at specified times)
            Q_list.append(
                (1 / (len(D_targets) + 1)) * (D_targets[i].T @ D_targets[i]))
        else:
            # otherwise it's zeros for everything
            print("zeros")
            Q_list.append(np.zeros((x.shape[0], x.shape[0])))
    # control acts on force
    # dynamic
    B = np.array(\
        [[0,0],\
         [0,0],\
         [0,0],\
         [0,0],\
         [1,0],\
         [0,1],\
         [0,0]])
    # cost
    R = r * np.eye(2) / len(times)

    return x, A, B, R, Q_list, noise_mag, Omega


def backward_recurse(A, B, R, Q, S):
    return A.T @ S @ A - (A.T @ S @ B) @ np.linalg.inv(R + B.T @ S @ B) @ (
        B.T @ S @ A) + Q


def compute_law(A, B, R, Q, S):
    return -np.linalg.inv(R + B.T @ S @ B) @ (B.T @ S @ A)


def compute_control(dt, x, A, B, R, Q_list, noise_mag, Omega):
    T = 1  # sec
    times = np.linspace(0, T, int(T / dt))

    # compute control law
    Q = Q_list[-1]
    S = Q_list[-1]
    S_list = [S]
    L_list = []
    for Q in Q_list[::-1]:  # no control at last time
        L = compute_law(A, B, R, Q, S)
        S = backward_recurse(A, B, R, Q, S)
        S_list.append(S)
        L_list.append(L)
    L_list = L_list[::-1]
    S_list = S_list[::-1]
    #     print(f"{len(S_list)} timesteps | {len(L_list)} control inputs")

    # main loop
    x_list = [x]
    u_list = []
    for i, t in enumerate(times):
        if i != len(times) - 1:
            # compute control
            u = L_list[i] @ x
            u_list.append(u)
        # run dynamic
        x = A @ x + B @ u + sample_state_noise(x, noise_mag, Omega)
        x_list.append(x.reshape(-1, 1))
    return x_list, u_list, L_list


def sample_APT_trajectories(num, noise_mag=0.01):

    trajectories = []
    control_trajectories = []
    control_law_trajectories = []

    # timesteps
    dt = 0.005  # sec
    T = .8  # sec
    times = np.linspace(0, T, int(T / dt))

    # hyperparams
    tau = 0.040  # 40ms, exponential filter constant
    m = .0005
    r = 0.002
    noise_mag = noise_mag
    wv = .01
    wf = 0

    for i in range(num):
        targets = generate_APT_targets(dt)
        x, A, B, R, Q_list, noise_mag, Omega = generate_dynamics(
            dt, targets, tau, m, r, noise_mag, wv, wf)
        x_list, u_list, L_list = compute_control(dt, x, A, B, R, Q_list,
                                                 noise_mag, Omega)
        trajectories.append(np.array(x_list).T[0])  ## HACK!!!
        control_trajectories.append(u_list)
        control_law_trajectories.append(L_list)

    return trajectories, control_trajectories, control_law_trajectories


def generate_arc(num_points, radius=250, start=0, end=180, polar=False):
    start_angle_rad = (np.pi / 180) * start
    end_angle_rad = (np.pi / 180) * end
    angles = np.linspace(end_angle_rad,
                         start_angle_rad,
                         num_points,
                         endpoint=True)
    radii = np.ones(num_points) * radius
    polar_coords = np.array(list(zip(radii, angles)))
    if polar:
        return polar_coords
    else:
        cartesian_coords = np.array(
            [polar_to_cartesian(p) for p in polar_coords])
        return cartesian_coords


def polar_to_cartesian(coord):
    return [coord[0] * np.cos(coord[1]), coord[0] * np.sin(coord[1])]


def cartesian_to_polar(coord):
    output = [0, 0]
    output[0] = np.sqrt(coord[0]**2 + coord[1]**2)
    # change behavior depending on quadrant
    output[1] = np.arccos(coord[0] / output[0])
    if coord[1] >= 0:
        output[1] = np.arccos(coord[0] / output[0])
    else:
        if coord[0] >= 0:
            output[1] = -np.arccos(coord[0] / output[0])
        else:
            output[1] = np.pi + np.arccos(-coord[0] / output[0])
    return output


def radians(degrees):
    return (np.pi / 180) * degrees


def make_polar_plot():
    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_subplot(projection="polar")
    ax.set_yticks([])
    #     ax.set_xlim([np.pi+np.pi/7,-np.pi/7])
    ax.set_ylim([0, 420])
    ax.set_xticks([0, np.pi])
    ax.set_xticklabels(["$0^{\circ}$", "$180^{\circ}$"], FontSize=18)
    ax.spines['polar'].set_visible(False)
    return fig, ax


trajectories, u_list, _ = sample_APT_trajectories(10)
trajectories = np.array(trajectories)

polar_trajectories = []
for t in trajectories:
    points = []
    for x in t.T:
        r, theta = cartesian_to_polar([x[0], x[1]])
        points.append([theta, r])
    polar_trajectories.append(np.stack(points).T)

fig, ax = make_polar_plot()
for u, pt in zip(u_list, polar_trajectories):
    plt.plot(pt[0], pt[1], alpha=0.1)

mean_curve = np.mean(np.array(polar_trajectories), axis=0)
print(mean_curve.shape)
std = np.std(np.array(polar_trajectories), axis=0)[1]
print(std.shape)
ax.fill_between(mean_curve[0],
                mean_curve[1] + std,
                mean_curve[1] - std,
                alpha=.5)
ax.plot(mean_curve[0], mean_curve[1], 'b', zorder=100)
arc = generate_arc(200, polar=True).T
ax.plot(arc[1], arc[0], "--k", zorder=100)

plt.figure()
for uvec, pvec in zip(u, trajectories[-1].T):
    plt.arrow(pvec[0], pvec[1], uvec[0][0], uvec[1][0])
    plt.plot(pvec[0], pvec[1], 'o', markerSize=0)

# plt.plot(std)
# print(std)

# trajectories, _, _ = sample_APT_trajectories(10, noise_mag=0.0075)
# trajectories = np.array(trajectories)

# print(trajectories.shape)

# polar_trajectories = []
# for t in trajectories:
#     points = []
#     for x in t.T:
#         r, theta = cartesian_to_polar([x[0], x[1]])
#         points.append([theta, r])
#     polar_trajectories.append(np.stack(points).T)

# print(np.array(polar_trajectories).shape)
# fig, ax = make_polar_plot()
# for pt in polar_trajectories:
#     ax.plot(pt[0], pt[1])
# ax.axis('off')
# ax.plot(arc[1], arc[0], "--k", zorder=100)

# trajectories, _, _ = sample_APT_trajectories(10, noise_mag=0.025)
# trajectories = np.array(trajectories)

# print(trajectories.shape)

# polar_trajectories = []
# for t in trajectories:
#     points = []
#     for x in t.T:
#         r, theta = cartesian_to_polar([x[0], x[1]])
#         points.append([theta, r])
#     polar_trajectories.append(np.stack(points).T)

# print(np.array(polar_trajectories).shape)
# fig, ax = make_polar_plot()
# for pt in polar_trajectories:
#     ax.plot(pt[0], pt[1])
# ax.axis('off')
# ax.plot(arc[1], arc[0], "--k", zorder=100)

# plt.plot(std)
# print(std)

# plt.figure(figsize=(18, 9))
# plt.xlabel("normalized time", FontSize=18)
# plt.ylabel("control signal", FontSize=18)
# plt.plot([u[0] for u in u_list[1:]], LineWidth=4)
# plt.plot([u[1] for u in u_list[1:]], LineWidth=4)
# plt.gca().set_xticks([0, 100, 200])
# plt.gca().set_xticklabels([0, 100, 200], FontSize=16)
# plt.gca().set_yticks([-100, 0, 100])
# plt.gca().set_yticklabels([-100, 0, 100], FontSize=16)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
# # plt.figure()
# # plt.title("control signal")
# # L_eigs = []
# # for L in L_list:
# #     L_eigs.append(np.sqrt(np.linalg.eig(L.T@L)[0][0]))
# # plt.plot(L_eigs)
# # plt.title("eigenvalues")

# ## velocity colored

# from matplotlib import cm
# fig = plt.figure(figsize=(15, 10))
# ax = fig.add_subplot(1, 1, 1)
# ax.axis('square')
# ax.set_xlim(-275, 400)
# ax.set_ylim(-10, 400)

# x_list = trajectories[0]
# print(np.array(x_list).shape)
# targets = generate_APT_targets(0.005)

# posx = [x[0] for x in x_list]
# posy = [x[1] for x in x_list]
# vel_mag = [np.sqrt(x[2]**2 + x[3]**2)[0] for x in x_list]

# steps = 10
# c = np.asarray(vel_mag)
# c -= np.min(c)
# c = c / np.max(c)
# it = 0

# while it < c.size - steps:
#     x_segm = posx[it:it + steps + 1]
#     y_segm = posy[it:it + steps + 1]
#     c_segm = plt.cm.jet(c[it + steps // 2])
#     ax.plot(x_segm, y_segm, c=c_segm)
#     it += steps

# ax.add_artist(plt.Circle((targets[0][0], targets[0][1]), 5, color='g'))
# for i, target in enumerate(targets[1:]):
#     ax.add_artist(
#         plt.Circle((target[0], target[1]), 0.5, color='k', alpha=0.25))
# ax.add_artist(plt.Circle((targets[-1][0], targets[-1][1]), 5, color='r'))
