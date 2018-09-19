import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from os import path
from PIL import Image

IMAGE_PATH = path.join(path.dirname(__file__), 'images')

# Maxmimum number of steps before game end
TIME_LIMIT = 512
MISS_LIMIT = 10

IMAGE_SIZE = 256

CURSOR_VALUE = 255
SHOT_VALUE = 0
CURSOR_SHOT_VALUE = 10

LOW_CLIP = 20
HIGH_CLIP = 234


class GullyAttackEnv(gym.Env):
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

    def __init__(self, images, image_path, time_limit=TIME_LIMIT, miss_limit=MISS_LIMIT, image_size=IMAGE_SIZE):
        self.images = images
        self.image_path = image_path
        self.image_size = image_size
        if not self._check_paths():
            raise ValueError("Invalid image paths, one or more images don't exist")

        self.TIME_LIMIT = time_limit
        self.MISS_LIMIT = miss_limit

        # The background and target mask is static 
        self.wallpaper = self._load_image_as_np(images['wallpaper'])
        self.background = self._load_image_as_np(images['background'])
        self.background[self.background > HIGH_CLIP] = HIGH_CLIP
        self.background[self.background < LOW_CLIP] = LOW_CLIP
        self.observation = None
        self.target = self._load_image_as_np(images['target'])

        # (Re)set all counters and state
        self.reset()

        # Check that the size is consistent
        if self.background.shape[0] != self.image_size or self.background.shape[1] != self.image_size:
            raise ValueError('Incorrect size of background image')
        if self.target.shape[0] != self.image_size or self.target.shape[1] != self.image_size:
            raise ValueError('Incorrect size of target image')

        # create an action space for moving and shooting (four directions, one for shooting)
        self._n_actions = 5
        self._action_space = spaces.Discrete(self._n_actions)

        self.viewer = None

    def _check_paths(self):
        print("background", path.join(self.image_path, self.images['background']))
        print("target", path.join(self.image_path, self.images['target']))
        return path.exists(path.join(self.image_path, self.images['background'])) and path.exists(path.join(self.image_path, self.images['target']))

    def _load_image_as_np(self, filename):
        img = Image.open(path.join(self.image_path, filename))
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
        # Increment time and check if game ends
        done = False
        self.time += 1
        if self.time > self.TIME_LIMIT:
            done = True

        reward = 0
        action_meaning = ACTION_MEANING[action]
        cursor_value = CURSOR_VALUE
        if action_meaning == 'UP':
            self.position[1] += 1
        elif action_meaning == 'DOWN':
            self.position[1] -= 1
        elif action_meaning == 'LEFT':
            self.position[0] -= 1
        elif action_meaning == 'RIGHT':
            self.position[0] += 1
        elif action_meaning == 'SHOOT':
            cursor_value = CURSOR_SHOT_VALUE
            # Check if previous shot
            if np.abs(self.hits[self.position[0], self.position[1]]) > 0.0:
                reward = -1
            # Check if new hit or miss
            else:
                if self.target[self.position[0], self.position[1]] > 0.0:
                    reward = 10
                    self.hits[self.position[0], self.position[1]] = 1
                else:
                    self.miss_count += 1
                    reward = -1
                    self.hits[self.position[0], self.position[1]] = -1
        
        if self.miss_count > self.MISS_LIMIT:
            reward = -100
            done = True

        # Enforce periodic boundary conditions to avoid moving out of frame
        self.position[0] = divmod(self.position[0], self.image_size)[1]
        self.position[1] = divmod(self.position[1], self.image_size)[1]

        # Assemble new observation (the background plus cursor and shot markers)
        self.observation = np.copy(self.background)
        # Mark your own position
        self.observation[self.position[0], self.position[1]] = cursor_value
        # Mark points you have shot at before 
        self.observation[np.abs(self.hits) > 0.0] = SHOT_VALUE

        return self.observation, reward, done, {}


    def reset(self):
        # Start in a random pixel in the image
        # self.position = [np.random.randint(low=0, high=self.image_size),
        #                           np.random.randint(low=0, high=self.image_size)]
        self.position = [int(self.image_size/2), int(self.image_size/2)]

        # Reset the non-static part of the observational space 
        self.hits = np.zeros_like(self.background, dtype=float)
        self.observation_space = np.dstack([self.background, self.hits]) 
        
        # Reset any counters 
        self.miss_count = 0
        self.time = 0
        
        self.observation = np.copy(self.background)
        self.observation[self.position[0], self.position[1]] = CURSOR_VALUE

        return self.observation

    def render(self, mode='human', close=False):
        if close and self.viewer:
            self.viewer.close()
            self.viewer = None
            return

        # background is an rgba-array. We use this for rendering.
        # img = np.copy(self.wallpaper[:, :, :-1])
        # img[self.position[0], self.position[1], :] = 0
        # hit_mask = self.hits > 0.0
        # img[hit_mask, :] = 0
        # miss_mask = self.hits < 0.0
        # img[miss_mask, 0] = 255        
        # img[miss_mask, 1] = 0        
        # img[miss_mask, 2] = 0        

        shape = self.observation.shape
        img = np.zeros((*shape, 3), dtype=np.uint8)
        img[:,:,0] = self.observation
        img[:,:,1] = self.observation
        img[:,:,2] = self.observation

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def close(self):
        pass

ACTION_MEANING = {
    0: "RIGHT",
    1: "LEFT",
    2: "UP",
    3: "DOWN",
    4: "SHOOT"
}