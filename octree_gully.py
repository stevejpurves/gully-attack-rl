import argparse
import time
import os
import gym
from octree_gully_env import OctreeEnv
import numpy as np
from os import path
from time import sleep

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

'''
    Arguments - setup this script to acept some arguments for training, testing and playback
'''
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='random')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()


# Setup the environment
IMAGE_SHAPE = (128, 128)
# Random seed
RANDOM_SEED = 43
# its possible as per the keras-rl atari example to preent multiple frames of teh environment
# to the network to allow teh network to see changes over a short time span
# setting this to 1 for the first pass, so agent will only make decisions based on the current state
NUM_FRAMES = 1
# the window applied to look back in sequential memory. We probably can do with no memory in this case
WINDOW_LENGTH = 1
# the background image (rgba) and the target
IMAGE_PATH = path.join(path.dirname(__file__), 'images')
IMAGES = {
    'wallpaper': 'rgb_il2300.crop.png',
    'background': 'grey_il2300.crop.png',
    'target': 'grey_label_il2300.crop.png'
}
# Folder to write weights and logs
LOG_FOLDER='logs'
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

np.random.seed(RANDOM_SEED)
env = OctreeEnv(image_size=IMAGE_SHAPE[0], 
                     images=IMAGES, 
                     image_path=IMAGE_PATH,
                     time_limit=1000,
                     miss_limit=100)
env.seed(RANDOM_SEED)

nb_actions = env.n_actions

###
#
#   Create the Model that will determine teh next action from the current environment

###
# This is the network from the keras-rl example
if args.mode == 'train' or args.mode == 'test':
    input_shape = (WINDOW_LENGTH,) + IMAGE_SHAPE
    model = Sequential()
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=input_shape))
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        model.add(Permute((1, 2, 3), input_shape=input_shape))

    model.add(Convolution2D(32, (8, 8), strides=(4, 4), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear')) # maybe this should be sigmoids?
    print(model.summary())

###
#
#   Select and Configure Agent
#
###
if args.mode == 'train' or args.mode == 'test':
    memory = SequentialMemory(limit=1000, window_length=WINDOW_LENGTH)
    #memory = EpisodeParameterMemory(limit=1000, window_length=1)

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
if args.mode == 'train' or args.mode == 'test':
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!
if args.mode == 'train' or args.mode == 'test':
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        nb_steps_warmup=100,
        gamma=.99,
        target_model_update=10000,
        train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])


# run this script based on arguments
if args.mode == 'train':
    print("start training")
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    timestamp = time.time()
    weights_filename = path.join(LOG_FOLDER, 'dqn_oc_{}_weights.h5f'.format(timestamp))
    checkpoint_weights_filename = path.join(LOG_FOLDER, 'dqn_oc_' + str(timestamp) + '_weights_{step}.h5f')
    log_filename = path.join(LOG_FOLDER, 'dqn_oc_{}_log.json'.format(timestamp))
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000, visualize=args.visualize)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
elif args.mode == 'test':
    print("start testing")
    if not args.weights:
        raise ValueError('weights filename must be specified when testing')
    weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=5, visualize=True)    
elif args.mode == 'random':
    print("random picker")
    env.reset()
    for n in range(4000):
        env.render()
        env.step(env.action_space.sample())
        sleep(0.1)