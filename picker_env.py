import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from os import path
from PIL import Image

IMAGE_PATH = path.join(path.dirname(__file__), 'images')

IMAGE_SIZE = 256
images = {
    'background': 'blob.png',
    'data': 'blob_grey.png',
    'reward': 'blob_edge.png'
}

class PickerEnv(gym.Env):
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

    def __init__(self):
        if not self._check_paths():
            raise ValueError("Invalid image paths, one or more images don't exist")

        self.background = self._load_image_as_np(images['background'])
        self.observation_space = self.background

        # reward - expect greyscale nd only use a single channel
        self.reward = self._load_image_as_np(images['reward'])[:,:,0]
        self.state = np.zeros_like(self.reward, dtype=np.uint8)
        assert self.state.shape[0] == IMAGE_SIZE and self.state.shape[1] == IMAGE_SIZE

        # create an action space containing image coordinates [i, j]
        self._n_actions = 2
        self._action_space = spaces.Dict({
            "i": spaces.Discrete(IMAGE_SIZE),
            "j": spaces.Discrete(IMAGE_SIZE)
        })

        self.viewer = None

    def _check_paths(self):
        print("background", path.join(IMAGE_PATH, images['background']))
        print("data", path.join(IMAGE_PATH, images['data']))
        print("reward", path.join(IMAGE_PATH, images['reward']))
        return path.exists(path.join(IMAGE_PATH, images['background'])) and path.exists(path.join(IMAGE_PATH, images['data'])) and path.exists(path.join(IMAGE_PATH, images['reward']))

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
        # action is always to pick a point, the action encodes the coordinates
        # if the action is outside the image, don't think this should happen
        if not self.action_space.contains(action):
            reward = -1
        else:
            # get a reward just for picking
            reward = 1 # action['i'] # or maybe reward picking higher up the image
            i = action['i']
            j = action['j']

            if self.state[i,j] == 1:
                # already picked this point
                reward = 0
            else:
                # this is a new pick
                reward = 1
                self.state[i,j] = 1
                self.observation_space[i,j,:] = 1

        return np.array(self.state), reward, done, {}


    def reset(self):
        self.observation_space = self.background
        self.state = np.zeros_like(self.reward)
        return np.array(self.state)

    def render(self, mode='human', close=False):
        if close and self.viewer:
            self.viewer.close()
            self.viewer = None
            return

        img = self.observation_space
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def close(self):
        pass