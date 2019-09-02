import numpy as np

from cs import CanonicalSystem

class DMPs(object):
    def __init__(self, n_dmps, n_bfs, dt=.01,
                 y0=0, goal=1, w=None, ay=None, by=None, **kwargs):
        '''
        :param n_dmps: int, number of dynamic motor primitives
        :param n_bfs: int, number of basis function per DMP
        :param dt: float, timestep of simulation
        :param y0: list, initial state of DMPs
        :param goal: list, goal state of DMPs
        :param w: list, control amplitude of basis functions
        :param ay: int, gain on attractor term y dynamic
        :param by: int, gain on attractor term y dynamic
        :param kwargs: param for dict data
        '''
        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.dt = dt

        if isinstance(y0, (int, float)):
            y0 = np.ones(n_dmps) * y0
        self.y0 = y0

        if isinstance(goal, (int, float)):
            goal = np.ones(n_dmps) * goal
        self.goal = goal

        if w is None:
            # default f = 0
            w = np.zeros((n_dmps, n_bfs))
        self.w = w

        self.ay = np.ones(n_dmps) * 25. if ay is None else ay
        self.by = self.ay / 4. if by is None else by

        # set up the cs
        self.cs = CanonicalSystem(self.dt, **kwargs)
        self.timesteps = int(self.cs.run_time / self.dt)

        # set up the DMP system
        self.reset_state()

    def check_offset(self):
        '''
        Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0
        '''

        for d in range(self.n_dmps):
            if(self.y0[d] == self.goal[d]):
                self.goal[0] += 1e-4

    def gen_front_term(self, x, dmp_num):
        # Generate locally weight regression param
        # return x * (self.goal[dmp_num] - self.y0[dmp_num])
        raise NotImplementedError()

    def gen_goal(self, y_des):
        # Generate the goal for path imitation
        # y_des: np.array, the desire trajectory to follow

        raise NotImplementedError()
        #return np.copy(y_des[:, -1])

    def gen_psi(self, x):
        # Generate the basis functions for a given canonical system rollout x
        # x: float, array: the canonical system state x or path
        raise NotImplementedError()
        # if isinstance(x, np.ndarray):
        #     x = x[:, None]
        # return np.exp(-self.h * (x - self.c)**2)

    def gen_weights(self, f_target):
        raise NotImplementedError()

    def imitate_path(self, y_des, plot=False):
        '''
        Generate weight w
        Takes in a desired trajectory and generate the set of system parameters that best realize this path
        :param y_des: list/array: the desired trajectories of each DMP, shape [n_dmps, run_time]
        :return: y_des
        '''

        # set initial state and goal, the desired goal
        if y_des.ndim == 1:
            y_des = y_des.reshape(1, len(y_des))
        self.y0 = y_des[:, 0].copy()
        self.y_des = y_des.copy()
        self.goal = self.gen_goal(y_des)

        self.check_offset()

        # generate function to interpolate the desired trajectory
        import scipy.interpolate
        path = np.zeros((self.n_dmps, self.timesteps))
        x = np.linspace(0, self.cs.run_time, y_des.shape[1])
        for d in range(self.n_dmps):
            path_gen = scipy.interpolate.interp1d(x, y_des[d])
            for t in range(self.timesteps):
                path[d, t] = path_gen(t * self.dt)
        y_des = path

        # calculate the velocity of y_des
        dy_des = np.diff(y_des) / self.dt
        # add zero to the beginning of every row
        dy_des = np.hstack((np.zeros((self.n_dmps, 1)), dy_des))

        # calculate the accelerate of y_des
        ddy_des = np.diff(dy_des) / self.dt
        # add zero to the beginning of every row
        ddy_des = np.hstack((np.zeros((self.n_dmps, 1)), ddy_des))

        # calculate f target function, shape [timesteps, n_dmps]
        f_target = np.zeros((y_des.shape[1], self.n_dmps))
        # find the force required to move along this trajectory
        for d in range(self.n_dmps):
            f_target[:, d] = (ddy_des[d] - self.ay[d] *
                              (self.by[d] * (self.goal[d] - y_des[d]) - dy_des[d]))

        # efficiency generate weight to realized f_target
        self.gen_weights(f_target)

        # if plot is True:

        self.reset_state()
        return y_des

    def rollout(self, timesteps=None, **kwargs):
        '''
        Run step for timesteps to generate a system trial
        :param timesteps:
        :param kwargs:
        :return: y_track, dy_track, ddy_track
        '''
        """Generate a system trial, no feedback is incorporated."""

        self.reset_state()

        if timesteps is None:
            if 'tau' in kwargs:
                timesteps = int(self.timesteps / kwargs['tau'])
            else:
                timesteps = self.timesteps

        # set up tracking vectors
        y_track = np.zeros((timesteps, self.n_dmps))
        dy_track = np.zeros((timesteps, self.n_dmps))
        ddy_track = np.zeros((timesteps, self.n_dmps))
        psi = np.zeros((timesteps, self.n_bfs))

        for t in range(timesteps):
            # run and record timestep
            y_track[t], dy_track[t], ddy_track[t], psi[t] = self.step(**kwargs)

        return y_track, dy_track, ddy_track, psi

    def reset_state(self):
        """Reset the system state"""
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()

    def step(self, tau=1.0, error=0.0, external_force=None):
        '''
        Run the DMP system for a single timestep.
        :param tau: float, scales the timestep, increase tau to make the system execute faster
        :param error: optional system feedback to adjust timestep
        :return: self.y, self.dy, self.ddy
        '''

        error_coupling = 1.0 / (1.0 + error)
        # run canonical system
        x = self.cs.step(tau=tau, error_coupling=error_coupling)

        # generate basis function activation
        psi = self.gen_psi(x)
        # print('psi',psi)

        for d in range(self.n_dmps):

            # generate forcing term
            f = (self.gen_front_term(x, d) * (np.dot(psi, self.w[d])) / np.sum(psi))

            # DMP acceleration
            # calculate y'' by f
            self.ddy[d] = (self.ay[d] *
                           (self.by[d] * (self.goal[d] -self.y[d]) -
                            self.dy[d] / tau) + f) * tau

            if external_force is not None:
                self.ddy[d] += external_force[d]

            self.dy[d] += self.ddy[d] * tau * self.dt * error_coupling
            self.y[d] += self.dy[d] * tau * self.dt * error_coupling

        return self.y, self.dy, self.ddy, psi