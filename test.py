import gym
import d4rl # Import required to register environments, you may need to also import the submodule
import matplotlib.pyplot as plt
import numpy as np
import wandb

# Create the environment
# env = gym.vector.make('antmaze-umaze-v0', reward_type="dense", num_envs=2)

# maze id
# antmaze-v1: two modes of up and bottom
# antmaze-v2: four goals
# antmaze-v2-hard: four goals with obstacles

run = wandb.init(project="diffusion_map", name="all")
names = ['antmaze-v1', 'antmaze-v2', 'antmaze-v2-hard', 'antmaze-v3', 'antmaze-v4', 'antmaze-v5',]
names = ['antmaze-v3']
for name in names:
    env = gym.make(name, reward_type="sparse")

    print("Observation space:", env.observation_space.shape)
    print("Action space:", env.action_space.shape)
    # d4rl abides by the OpenAI gym interface

    obs = env.reset()
    for i in range(1):
        img = env.render(mode="rgb_array")
        img = wandb.Image(img)
        wandb.log({'map': img})
    # plt.imshow(img)
    # plt.pause(1/5)
    # plt.clf()
    # print(env.get_xy())
        obs, r, done, info = env.step(env.action_space.sample())
    # print(i, r, done, info)
    # if done:
    #   env.reset()
    # if i == 0:
    #   import time
    #   time.sleep(100)
