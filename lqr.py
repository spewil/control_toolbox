import sys
import numpy as np
from loguru import logger

fmt = "{time} - {name} - {level} - {message}"
logger.add("debug.log", level="DEBUG", format=fmt)
logger.add(sys.stderr, level="ERROR", format=fmt)


class LQR():
    def __init__(self, dynamic_params, cost_params):
        self.target = cost_params['target']
        self.dynamics_params = dynamic_params
        self.cost_params = cost_params
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
        A = np.eye(state_dim + 2)
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
        B = np.eye(state_dim + 2)
        B = B[:, 4:6]

        return A, B

    def compute_costs(self):

        dt = self.cost_params["dt"]
        target = self.target
        velocity_cost_amplitude = self.cost_params["velocity_cost_amplitude"]
        force_cost_amplitude = self.cost_params["force_cost_amplitude"]
        control_cost_magnitude = self.cost_params["control_cost_magnitude"]

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
        w = np.append(state_noise, np.zeros((2, 1)), axis=0)
        x = self.A @ x + self.B @ u + w
        return x, u

    def sample_state_noise(self):
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
        state_noise_mean = np.zeros(self.dynamics_params["state_dim"])
        return state_noise_magnitude * np.random.default_rng(
        ).multivariate_normal(state_noise_mean,
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
            state_noise = self.sample_state_noise()
            if perturbation is not None:
                x_in = np.add(perturbation[:, i].reshape(-1, 1), x_list[i])
            else:
                x_in = x_list[i]
            x, u = self.advance_dynamic(x_in, state_noise)
            x_list.append(x.reshape(-1, 1))
            if i != steps - 1:
                u_list.append(u.reshape(-1, 1))
        return np.array(x_list)[:, :, 0], np.array(u_list)[:, :, 0]

    def backward_pass(self):
        """
            For finite time control and cost-to-go list
        """
        A = self.A
        B = self.B
        R = self.R
        Q = self.Q
        S = Q
        S_list = [S]
        L_list = []
        L = self.compute_control_law(A, B, R, Q, S)
        S = self.backward_recurse(A, B, R, Q, S)
        S_list.append(S)
        L_list.append(L)
        L_list = L_list[::-1]
        S_list = S_list[::-1]
        return np.array(L_list), np.array(S_list)


def sample_trajectories(lqr,
                        num_movements,
                        num_steps,
                        lims=[-500, 500],
                        perturbation=None):
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