# Gully Attack

The RL portion of the work done at the Force ML Hackathon 2018

This setup has been created from merging the [`keras-rl` atari example](https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py) with a custom environment using images as state and a basis for rewards.

check the [pitch deck](the_reinforcers_gully_attack.pdf) for details of the project

## Installation

 - pip install gym
 - pip install keras-rls

## running

 - python run_gully_attack.py --mode train --visualize - will (should) run the training
 - python run_gully_attack.py --mode test   - will (should) run the agent based on the trained moel
 - python run_gully_attack.py random - just run the environment and draw random dotss 