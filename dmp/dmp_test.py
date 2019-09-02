from dmp import DMPs
import numpy as np
import readfile as rf

class DMPs_discrete(DMPs):
    '''
    An implementation of discrete DMPs
    '''

    def __init__(self, **kwargs):
        # call super class constructor
        super(DMPs_discrete, self).__init__(pattern='discrete', **kwargs)

        self.gen_centers()

        # set variance (h) of Gaussian basis functions
        # trial and error to find this spacing???
        self.h = np.ones(self.n_bfs) * self.n_bfs**1.5 / self.c / self.cs.ax

        self.check_offset()

    def gen_centers(self):
        '''
        Calculate ci
        Set the center of the Gaussian basis
        function be spaced evenly throughout run time
        :return:
        '''

        # desired activations throughout time
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.ones(len(des_c))

        for n in range(len(des_c)):
            self.c[n] = np.exp(-self.cs.ax * des_c[n])

    def gen_front_term(self, x, dmp_num):
        '''
        Generates the diminishing front term on the force term.
        :param x: float, the current value of the canonical system
        :param dmp_num: int, the index of the current dmp
        :return: x(g-y0)
        '''
        return x * (self.goal[dmp_num] - self.y0[dmp_num])

    def gen_goal(self, y_des):
        '''
        Generate the goal for path imitation
        :param y_des: np.array, the desired trajectory to follow
        :return:
        '''
        return np.copy(self.y_des[:, -1])

    def gen_psi(self, x):
        '''?????????????????????????? x = x[:, None] what????????
        Generate the basis functions for a given canonical system rollout x
        :param x:
        :return: phy function
        '''
        if isinstance(x, np.ndarray):
            x = x[:, None]
        return np.exp(-self.h * (x - self.c)**2)

    def gen_weights(self, f_target):
        '''
        Generate a set of weight over the basis functions
        such that the target forcing term trajectory is matched
        :param f_target:
        '''

        # calculate x and psi
        x_track = self.cs.rollout()
        psi_track = self.gen_psi(x_track)

        # calculate BF weights using weighted linear regression
        self.w = np.zeros((self.n_dmps, self.n_bfs))
        for d in range(self.n_dmps):
            # spatial scaling term
            k = (self.goal[d] - self.y0[d])
            for b in range(self.n_bfs):
                numer = np.sum(x_track * psi_track[:, b] * f_target[:, d] )
                denom = np.sum(x_track**2 * psi_track[:, b])
                self.w[d, b] = numer / (k * denom)
        self.w = np.nan_to_num(self.w)

###############################
#########Test code#############
###############################
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # files=['2']
    # trajectory = rf.get_sequence(files, writebox=[-1,1,0,1], spaces=False)
    x_trajectory = [-1.5500e-1, -1.5600e-1, -1.5700e-1, -1.5800e-1, -1.5900e-1, -1.6000e-1,
    -1.6100e-1, -1.6200e-1, -1.6300e-1, -1.6400e-1]
    y_trajectory = [+0.0000e+0, +0.0000e+0, +0.0000e+0, +0.0000e+0, +0.0000e+0, +0.0000e+0,
    +0.0000e+0, +0.0000e+0, +0.0000e+0, +0.0000e+0]
    z_trajectory = [+6.7500e-1, +6.7500e-1, +6.7500e-1, +6.7500e-1, +6.7500e-1, +6.7500e-1,
    +6.7500e-1, +6.7500e-1, +6.7500e-1, +6.7500e-1]
    trajectory=np.array([x_trajectory, y_trajectory, z_trajectory])
    print('trajectory', trajectory)
    # num_DOF = 2
    num_DOF = trajectory.shape[0]
    print('num_DOF', num_DOF)
    # break up the trajectory into its different words
    # NaN or None signals a new word / break in drawing
    # breaks = np.array(np.where(trajectory[0] != trajectory[0]))[0]
    # num_seqs = len(breaks) - 1
    # print('num_seqs', num_seqs)
    # # num_seqs is the number of characters
    # dmp_sets = []
    # for ii in range(num_seqs):
    #     # get the ii'th sequence
    #     seq = trajectory[:, breaks[ii]+1:breaks[ii+1]]
    #     dmps = DMPs_discrete(n_dmps=num_DOF, n_bfs=100)

    #     dmps.imitate_path(y_des=seq)
    #     dmp_sets.append(dmps)
    dmps = DMPs_discrete(n_dmps=num_DOF, n_bfs=100)

    dmps.imitate_path(y_des=trajectory)
    y_track, dy_track, ddy_track = dmps.rollout()
    print("y_track",y_track)

    # plt.figure(2)
    # plt.plot(y_track[:, 0], y_track[:, 1], lw=2)
    # a = plt.plot(trajectory[0,:], trajectory[1,:], 'r--', lw=2)
    # plt.legend([a[0]], ['desired path'], loc='lower right')
    # plt.tight_layout()
    # plt.show()

    




    # # test imitation of path run
    # plt.figure(2, figsize=(6, 4))
    # n_bfs = [10, 30, 50, 100, 10000]

    # # a straight line to target
    # path1 = np.sin(np.arange(0, 1, .01)*5)
    # # a strange path to target
    # path2 = np.zeros(path1.shape)
    # path2[int(len(path2) / 2.):] = .5

    # for ii, bfs in enumerate(n_bfs):
    #     dmp = DMPs_discrete(n_dmps=2, n_bfs=bfs)

    #     dmp.imitate_path(y_des=np.array([path1, path2]))
    #     # change the scale of the movement
    #     dmp.goal[0] = 3
    #     dmp.goal[1] = 2

    #     y_track, dy_track, ddy_track = dmp.rollout()

    #     plt.figure(2)
    #     plt.subplot(211)
    #     plt.plot(y_track[:, 0], lw=2)
    #     # plt.plot(y_track[:, 0], y_track[:, 1], lw=2)
    #     plt.subplot(212)
    #     plt.plot(y_track[:, 1], lw=2)

    # plt.subplot(211)
    # a = plt.plot(path1 / path1[-1] * dmp.goal[0], 'r--', lw=2)
    # plt.title('DMP imitate path 1')
    # plt.xlabel('time (ms)')
    # plt.ylabel('system trajectory')
    # plt.legend([a[0]], ['desired path'], loc='lower right')
    # plt.subplot(212)
    # b = plt.plot(path2 / path2[-1] * dmp.goal[1], 'r--', lw=2)
    # plt.title('DMP imitate path 2')
    # plt.xlabel('time (ms)')
    # plt.ylabel('system trajectory')
    # plt.legend(['%i BFs' % i for i in n_bfs], loc='lower right')

    # plt.tight_layout()
    # plt.show()