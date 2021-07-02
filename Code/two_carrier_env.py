import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
import gym
from gym import spaces
from gym.utils import seeding
# import matplotlib
# matplotlib.use( 'tkagg' )

# This is a customized version. Original source of the gym environment is available below. 
# https://github.com/irasatuc/gym-ae5117

class TwoCarrierEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.seed()
        self.viewer = None
        self.prev_reward = None
        self.max_episode_steps = 1000
        self.observation_space = spaces.Box(low=-10., high=10., shape=(3,), dtype=np.float32)
        self.action_space = spaces.Tuple((spaces.Discrete(4), spaces.Discrete(4))) # ^v<>
        self.action_codebook = np.array([
            [0., .02],
            [0., -.02],
            [-.02, 0.],
            [.02, 0.]
        ])
        # vars
        self.rod_pose = np.zeros(3)
        self.c0_position = np.array([
            self.rod_pose[0]+.5*np.cos(self.rod_pose[-1]), 
            self.rod_pose[1]+.5*np.sin(self.rod_pose[-1]),
            0
        ])
        self.c1_position = np.array([
            self.rod_pose[0]-.5*np.cos(self.rod_pose[-1]), 
            self.rod_pose[1]-.5*np.sin(self.rod_pose[-1]),
            0
        ])
        self.c0_traj = []
        self.c1_traj = []
        # prepare renderer
        self.fig = plt.figure(figsize=(12,8))
        self.ax = self.fig.add_subplot(111)
        # nwwpat = Rectangle(xy=(-5.5,5), width=5.1, height=.5, fc='gray')
        # newpat = Rectangle(xy=(.4,5), width=5.1, height=.5, fc='gray')
        # wwpat = Rectangle(xy=(-5.5,-.5), width=.5, height=6, fc='gray')
        # ewpat = Rectangle(xy=(5,-.5), width=.5, height=6, fc='gray')
        # swpat = Rectangle(xy=(-5.5,-.5), width=11, height=.5, fc='gray')

        # Environment walls
        nwwpat = Rectangle(xy=(-2.5,3), width=2.1, height=.5, fc='gray')
        newpat = Rectangle(xy=(.4,3), width=2.1, height=.5, fc='gray')
        wwpat = Rectangle(xy=(-2.5,-.5), width=.5, height=4, fc='gray')
        ewpat = Rectangle(xy=(2,-.5), width=.5, height=4, fc='gray')
        swpat = Rectangle(xy=(-2.5,-.5), width=5, height=.5, fc='gray')
        self.fixed_patches = [nwwpat, newpat, wwpat, ewpat, swpat]
            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_counter = 0
        # initial rod coordinates
        x = np.random.uniform(-1.4, 1.4)
        y, theta = (0.2, 0.) if np.random.random() < 0.07 else (np.random.uniform(0.6, 2.4), np.random.uniform(-1.57, 1.57))

        self.rod_pose = np.array([x, y, theta])
        self.c0_position = np.array([
            self.rod_pose[0]+.5*np.cos(self.rod_pose[-1]), 
            self.rod_pose[1]+.5*np.sin(self.rod_pose[-1]),
            theta
        ])
        self.c1_position = np.array([
            self.rod_pose[0]-.5*np.cos(self.rod_pose[-1]), 
            self.rod_pose[1]-.5*np.sin(self.rod_pose[-1]),
            theta
        ])
        self.c0_traj = [self.c0_position.copy()]
        self.c1_traj = [self.c1_position.copy()]

        return self.c0_position, self.c1_position
        
    def step(self, action):
        done = False
        info = ''
        reward = 0
        
        prev_c0 = self.c0_position.copy()
        prev_c1 = self.c1_position.copy()
        
        # compute rod's displacement and rotation
        disp = self.action_codebook[action[0]] + self.action_codebook[action[1]]
        rot = 0.
        rot += -np.arctan2(self.action_codebook[action[0]][0]*np.sin(self.rod_pose[-1]), .5) + \
            np.arctan2(self.action_codebook[action[0]][1]*np.cos(self.rod_pose[-1]), .5) + \
            np.arctan2(self.action_codebook[action[1]][0]*np.sin(self.rod_pose[-1]), .5) - \
            np.arctan2(self.action_codebook[action[1]][1]*np.cos(self.rod_pose[-1]), .5)
        deltas = np.append(disp, rot)
        self.rod_pose += deltas
        self.c0_position = np.array([
            self.rod_pose[0]+.5*np.cos(self.rod_pose[-1]), 
            self.rod_pose[1]+.5*np.sin(self.rod_pose[-1]),
            self.rod_pose[2]
        ])
        self.c1_position = np.array([
            self.rod_pose[0]-.5*np.cos(self.rod_pose[-1]), 
            self.rod_pose[1]-.5*np.sin(self.rod_pose[-1]),
            self.rod_pose[2]
        ])
        self.c0_traj.append(self.c0_position.copy())
        self.c1_traj.append(self.c1_position.copy())
        
        # restrict angle in (-pi,pi)
        if np.pi<self.rod_pose[-1]<=2*np.pi:
            self.rod_pose[-1] -= 2*np.pi 
        elif -np.pi>self.rod_pose[-1]>=-2*np.pi:
            self.rod_pose[-1] += 2*np.pi
            
        # compute reward
        reward = (np.abs(prev_c0[0])-np.abs(self.c0_position[0]) + np.abs(prev_c1[0])-np.abs(self.c1_position[0]) + \
            (self.c0_position[1]-prev_c0[1] + self.c1_position[1]-prev_c1[1]))
        
        # check crash
        rod_points = np.linspace(self.c0_position[0:2], self.c1_position[0:2], 50)
        for p in self.fixed_patches:
            if np.sum(p.contains_points(rod_points, radius=.001)):
                done = True
                info = 'crash wall'
                reward = -40
                break
        # check escape
        if self.c0_position[1]>3.5 and self.c1_position[1]>3.5:
            reward = 200.
            done = True
            info = 'escaped'

        return self.c0_position, self.c1_position, reward, done, info

    def render(self, mode='human'):
        self.ax = self.fig.get_axes()[0]
        self.ax.cla()
        patch_list = []
        patch_list += self.fixed_patches
        # add wall patches
        c0pat = Circle(
            xy=(self.c0_position[0], self.c0_position[1]), 
            radius=.05, 
            ec='black',
            fc='white'
        )
        patch_list.append(c0pat)
        c1pat = Circle(
            xy=(self.c1_position[0], self.c1_position[1]), 
            radius=.05, 
            fc='black'
        )
        patch_list.append(c1pat)
        pc = PatchCollection(patch_list, match_original=True) # match_origin prevent PatchCollection mess up original color
        # plot patches
        self.ax.add_collection(pc)
        # plot rod
        self.ax.plot(
            [self.c0_position[0], self.c1_position[0]],
            [self.c0_position[1], self.c1_position[1]],
            color='darkorange'
        )
        # plot trajectory
        if self.c0_traj and self.c0_traj:
            traj_c0 = np.array(self.c0_traj)
            traj_c1 = np.array(self.c1_traj)
            self.ax.plot(traj_c0[-100:,0], traj_c0[-100:,1], linestyle=':', linewidth=0.5, color='black')
            self.ax.plot(traj_c1[-100:,0], traj_c1[-100:,1], linestyle=':', linewidth=0.5, color='black')
        # Set ax
        self.ax.axis(np.array([-3, 3, -1, 4.5]))
        self.ax.set_xticks(np.arange(-3, 3))
        self.ax.set_yticks(np.arange(-1, 5))
        self.ax.grid(color='grey', linestyle=':', linewidth=0.5)
        plt.pause(0.02)
        self.fig.show()

# env = TwoCarrierEnv()
# env.render()
