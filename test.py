import gym
import d4rl # Import required to register environments, you may need to also import the submodule
import matplotlib.pyplot as plt
import numpy as np
# Create the environment
# env = gym.vector.make('antmaze-umaze-v0', reward_type="dense", num_envs=2)
env = gym.make('antmaze-umaze-v0', reward_type="dense")

print(env.observation_space.shape, env.action_space.shape)
# d4rl abides by the OpenAI gym interface
obs = env.reset()
for i in range(1000):
    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.pause(1/5)
    plt.clf()
    # print(env.get_xy())
    obs, _, done, _ = env.step(env.action_space.sample())
    # print(i, done)
    # if done:
    #   env.reset()
    # if i == 0:
    #   import time
    #   time.sleep(100)