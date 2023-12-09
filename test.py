import gym
import d4rl # Import required to register environments, you may need to also import the submodule
import matplotlib.pyplot as plt
import numpy as np
import wandb

# Create the environment
env = gym.vector.make('antmaze-v1', reward_type="dense", num_envs=2)
print(env.__dict__)
assert 0
# maze id
# antmaze-v1: two modes of up and bottom
# antmaze-v2: four goals
# antmaze-v2-hard: four goals with obstacles

env = gym.make('antmaze-v1', reward_type="sparse")
print(env.env.env.spec.kwargs['maze_map'])

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def plot_density(kwargs):
    # x = map[:,0]
    # y = map[:,1]
    
    maze_map = kwargs['maze_map']
    maze_size = kwargs['maze_size_scaling']

    start = None
    goals = []
    blocks = []
    # find start and goal positions
    for i in range(len(maze_map)):
        for j in range(len(maze_map[0])):
            if maze_map[i][j] == 'r':
                start = (i, j)
            elif maze_map[i][j] == 'g':
                goals.append((i, j))
            elif maze_map[i][j] == 1:
                blocks.append((i, j))
    
    fig, ax = plt.subplots()

    # compute limit
    x_lim = (-(start[1] + 0.5) * maze_size, (len(maze_map[0]) - 0.5 - start[1]) * maze_size)
    y_lim = (-(len(maze_map[0]) - 0.5 - start[0]) * maze_size, (start[0] + 0.5) * maze_size)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # draw blocks
    for block in blocks:
        x, y = x_lim[0] + maze_size * block[1], maze_size * block[0] - y_lim[1]
        ax.add_patch(Rectangle((x, y), maze_size, maze_size))
    # sns.kdeplot(x=x, y=y, cmap="Reds", shade=True, bw_adjust=.5)

    # draw start and goal positions
    ax.plot(0, 0, 'ro')
    ax.annotate('start', (0, 0.25))
    for goal in goals:
        x = (goal[1] - start[1]) * maze_size
        y = (goal[0] - start[0]) * maze_size
        ax.plot(x, y, 'bo')
        ax.annotate('goal', (x, y + 0.25))
    # plt.show()
    # plt.savefig(f'dist_density/{name}.png')

    fig.canvas.draw()  # Draw the canvas, cache the renderer
    image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # reversed converts (W, H) from get_width_height to (H, W)
    image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
    return image

#img = plot_density(env.env.env.spec.kwargs)

run = wandb.init(project="diffusion_map", name="all")
#img = wandb.Image(img)
#wandb.log({'map': img})
#assert 0

names = ['antmaze-v1', 'antmaze-v2', 'antmaze-v2-hard', 'antmaze-v3', 'antmaze-v4', 'antmaze-v5',]
#names = ['antmaze-v3']
for name in names:
    env = gym.make(name, reward_type="sparse")

    print("Observation space:", env.observation_space.shape)
    print("Action space:", env.action_space.shape)
    # d4rl abides by the OpenAI gym interface

    obs = env.reset()
    for i in range(1):
        img = env.render(mode="rgb_array")
        img = wandb.Image(img)
        img2 = plot_density(env.env.env.spec.kwargs)
        img2 = wandb.Image(img2)
        wandb.log({'original': img, 'draw': img2})
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
