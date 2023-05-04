import matplotlib.pyplot as plt
import numpy as np

def valid_node(node, size_of_grid):
    """Checks if node is within the grid boundaries."""
    if node[0] < 0 or node[0] >= size_of_grid:
        return False
    if node[1] < 0 or node[1] >= size_of_grid:
        return False
    return True

def up(node):
    return (node[0]-1,node[1])

def down(node):
    return (node[0]+1,node[1])

def left(node):
    return (node[0],node[1]-1)

def right(node):
    return (node[0],node[1]+1)

def backtrack(initial_node, desired_node, distances):
    # idea start at the last node then choose the least number of steps to go back
    # last node
    path = [desired_node]

    size_of_grid = distances.shape[0]

    while True:
        # check up down left right - choose the direction that has the least distance
        potential_distances = []
        potential_nodes = []

        directions = [up,down,left,right]

        for direction in directions:
            node = direction(path[-1])
            if valid_node(node, size_of_grid):
                potential_nodes.append(node)
                potential_distances.append(distances[node[0],node[1]])

        least_distance_index = np.argmin(potential_distances)
        path.append(potential_nodes[least_distance_index])

        if path[-1][0] == initial_node[0] and path[-1][1] == initial_node[1]:
            break

    return list(reversed(path))

def dijkstra(initial_node, desired_node, obstacles):
    """Dijkstras algorithm for finding the shortest path between two nodes in a graph.

    Args:
        initial_node (list): [row,col] coordinates of the initial node
        desired_node (list): [row,col] coordinates of the desired node
        obstacles (array 2d): 2d numpy array that contains any obstacles as 1s and free space as 0s

    Returns:
        list[list]: list of list of nodes that form the shortest path
    """
    # initialize cost heuristic map
    obstacles = obstacles.copy()
    # obstacles should have very high cost, so we avoid them.
    obstacles *= 1000
    # normal tiles should have 1 cost (1 so we can backtrack)
    obstacles += np.ones(obstacles.shape)
    # source and destination are free
    obstacles[initial_node[0],initial_node[1]] = 0
    obstacles[desired_node[0],desired_node[1]] = 0


    # initialize maps for distances and visited nodes
    size_of_floor = obstacles.shape[0]

    # we only want to visit nodes once
    visited = np.zeros([size_of_floor,size_of_floor],bool)

    # initiate matrix to keep track of distance to source node
    # initial distance to nodes is infinity so we always get a lower actual distance
    distances = np.ones([size_of_floor,size_of_floor]) * np.inf
    # initial node has a distance of 0 to itself
    distances[initial_node[0],initial_node[1]] = 0

    # start algorithm
    current_node = [initial_node[0], initial_node[1]]
    while True:
        directions = [up, down, left, right]
        for direction in directions:
            potential_node = direction(current_node)
            if valid_node(potential_node, size_of_floor): # boundary checking
                if not visited[potential_node[0],potential_node[1]]: # check if we have visited this node before
                    # update distance to node
                    distance = distances[current_node[0], current_node[1]] + obstacles[potential_node[0],potential_node[1]]

                    # update distance if it is the shortest discovered
                    if distance < distances[potential_node[0],potential_node[1]]:
                        distances[potential_node[0],potential_node[1]] = distance


        # mark current node as visited
        visited[current_node[0]  ,current_node[1]] = True

        # select next node
        # by choosing the the shortest path so far
        t=distances.copy()
        # we don't want to visit nodes that have already been visited
        t[np.where(visited)]=np.inf
        # choose the shortest path
        node_index = np.argmin(t)

        # convert index to row,col.
        node_row = node_index//size_of_floor
        node_col = node_index%size_of_floor
        # update current node.
        current_node = (node_row, node_col)

        # stop if we have reached the desired node
        if current_node[0] == desired_node[0] and current_node[1]==desired_node[1]:
            break

    # backtrack to construct path
    return backtrack(initial_node,desired_node,distances)
  

def generate_obstacles(map, resolution):
    scaled_map = np.zeros((resolution, resolution))
    assert resolution % len(map) == 0
    scale = resolution // len(map)
    for i in range(len(map)):
        for j in range(len(map[i])):
          if map[i][j] == G:
            goal = (i, j)
            continue
          elif map[i][j] == 0:
            continue
          elif map[i][j] == R:
            reset = (i, j)
            continue
          else:
            for n in range(scale):
              for m in range(scale): 
                if map[i][j] == 1:
                  scaled_map[i*scale+m][j*scale+n] = 1
    goal = [goal[0]*scale+scale//2, goal[1]*scale+scale//2]
    reset = [reset[0]*scale+scale//2, reset[1]*scale+scale//2]
    return scaled_map, goal, reset
    
    
def generate_reward_map(map, resolution):
  map, goal, reset = generate_obstacles(map, resolution)
  reward_map = np.zeros_like(map)
  for i in range(map.shape[0]):
    print(i)
    for j in range(map.shape[1]):
      if map[i][j] == 1:
        reward_map[i][j] == -10001
      elif [i,j] == goal:
        reward_map[i][j] = 0
      else:
        reward_map[i][j] = -len(dijkstra([i,j], goal, map))
  return reward_map
  
  
RESET = R = 'r'  # Reset position.
GOAL = G = 'g'
    
U_MAZE_TEST = [[1, 1, 1, 1, 1],
              [1, 0, 0, 0, 1],
              [1, G, 1, R, 1],
              [1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1]]

# obstacles, goal, reset = generate_obstacles(U_MAZE_TEST, resolution=100)

# obstacles = np.array([[0,0,0,0,0,0,0,0,0,0],
#                       [0,0,0,0,0,0,0,0,0,0],
#                       [0,0,0,0,1,0,0,0,0,0],
#                       [0,0,0,1,0,1,0,0,0,0],
#                       [1,1,1,1,0,1,1,0,0,0],
#                       [0,0,0,1,0,1,0,0,0,0],
#                       [0,0,0,0,0,1,1,0,0,0],
#                       [0,0,0,0,0,1,0,0,0,0],
#                       [0,0,0,0,1,0,0,0,0,0],
#                       [0,0,0,0,0,0,0,0,0,0]], dtype=float)
# print(reset)
import time
start = time.time()
reward_map = generate_reward_map(U_MAZE_TEST, resolution=60)
with open('/data/pulkitag/misc/steven/D4RL/reward_map.npy', 'wb') as f:
  np.save(f, reward_map)
# path = dijkstra([51,70],goal,obstacles)
print(time.time()-start)

    
# p = np.zeros(shape=obstacles.shape)
# for i in range(len(path)):
#     p[path[i][0],path[i][1]] = np.nan

# plt.imshow(p+obstacles, cmap='jet')
# plt.show()