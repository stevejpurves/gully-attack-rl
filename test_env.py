import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from os import path
from PIL import Image

IMAGE_PATH = path.join(path.dirname(__file__), 'images')

IMAGE_SIZE = 128
images = {
    'wallpaper': 'rgb_il2300.crop.png',
    'background': 'grey_il2300.crop.png',
    'result': 'grey_label_il2300.crop.png',
}

class TestEnv(gym.Env):
    """The main OpenAI Gym class. It encapsulates an environment with
        arbitrary behind-the-scenes dynamics. An environment can be
        partially or fully observed.
        The main API methods that users of this class need to know are:
            step
            reset
            render
            close
            seed
        And set the following attributes:
            action_space: The Space object corresponding to valid actions
            observation_space: The Space object corresponding to valid observations
            reward_range: A tuple corresponding to the min and max possible rewards
        Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
        The methods are accessed publicly as "step", "reset", etc.. The
        non-underscored versions are wrapper methods to which we may add
        functionality over time.
    """

    def __init__(self, image_power=7):
        if not self._check_paths():
            raise ValueError("Invalid image paths, one or more images don't exist")

        # Load images
        self.background = self._load_image_as_np(images['background'])
        self.wallpaper = self._load_image_as_np(images['wallpaper'])
        self.result = self._load_image_as_np(images['result'])
        self.state = np.zeros_like(self.result)

        # Observation space
        self.observation_space = self.background

        # Action space
        self._n_actions = 4
        self._action_space = spaces.Discrete(self.n_actions)
        self.level = 0
        self.position = np.zeros(image_power)
        self.image_power = image_power
        self.image_size = 2**image_power

        # Reward range
        #self.reward_range = (-1,1)

        self.viewer = None

    def _check_paths(self):
        print("background", path.join(IMAGE_PATH, images['background']))
        print("wallpaper", path.join(IMAGE_PATH, images['wallpaper']))
        print("result", path.join(IMAGE_PATH, images['result']))
        return path.exists(path.join(IMAGE_PATH, images['background'])) and path.exists(path.join(IMAGE_PATH, images['result'])) and path.exists(path.join(IMAGE_PATH, images['wallpaper']))

    def _load_image_as_np(self, filename):
        img = Image.open(path.join(IMAGE_PATH, filename))
        return np.array(img)

    @property
    def action_space(self):
        return self._action_space

    @property
    def n_actions(self):
        return self._n_actions

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
            episode is reached, you are responsible for calling `reset()`
            to reset this environment's state.
            Accepts an action and returns a tuple (observation, reward, done, info).
            Args:
                action (object): an action provided by the environment
            Returns:
                observation (object): agent's observation of the current environment
                reward (float) : amount of reward returned after previous action
                done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
                info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        done = False
        reward = 0

        self.position[self.level] = action
        self.level += 1

        if self.level >= self.image_power-1:
            offx = 0
            offy = 0
            deltax = self.image_size
            deltay = self.image_size
            for choice in self.position:
                deltax = deltax//2
                deltay = deltay//2
                if choice == 0:
                    pass
                elif choice == 1:
                    offx += deltax
                elif choice == 2:
                    offy += deltay
                elif choice == 3:
                    offx += deltax
                    offy += deltay
            val = self.state[offx, offy]
            if val == 0:
                if self.result[offx, offy] == 0:
                    reward = 1
                else:
                    reward = -1
                self.state[offx, offy] = reward
            else:
                reward == -1
            self.level = 0
            self.position.fill(0)
        return np.array(self.state), reward, done, {}


    def reset(self):
        self.observation_space = self.background
        self.state = np.zeros_like(self.result)
        return np.array(self.state)

    def render(self, mode='human', close=False):
        if close and self.viewer:
            self.viewer.close()
            self.viewer = None
            return

        img = self.wallpaper
        #ps = (self.state!=0)*255
        #img = img + np.stack((ps, ps, ps, ps), axis=2)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def close(self):
        pass