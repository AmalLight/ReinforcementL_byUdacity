
import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

# Set plotting options
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)

get_ipython().system('python -m pip install pyvirtualdisplay')
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

is_ipython = 'inline' in plt.get_backend()
if is_ipython: from IPython import display

plt.ion()

# Create an environment and set random seed
env = gym.make('MountainCar-v0')
env.seed(505);

# -------------------------------------------------------

# Explore state (observation) space
print("State space:", env.observation_space)
print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)

# Generate some samples from the state space 
print("State space samples:")
print(np.array([env.observation_space.sample() for i in range(10)]))

# Explore the action space
print("Action space:", env.action_space)

# Generate some samples from the action space
print("Action space samples:")
print(np.array([env.action_space.sample() for i in range(10)]))

# -------------------------------------------------------


def create_uniform_grid(low, high, bins=(10, 10)):
    """
    Parameters:
       low : array_like
       high : array_like
       bins : tuple
    Returns: grid : list of array
    """
    assert bins [ 0 ] == bins [ 1 ]

    x =  low [ 0 ] - high [ 0 ]
    if x < 0 : x *= -1
    x /=  bins [ 0 ]
    x = round ( x , 3 )
    print ( 'x:' , x )

    y = low [ 1 ] - high [ 1 ]
    if y < 0 : y *= -1
    y /=  bins [ 1 ]
    y = round ( y , 3 )
    print ( 'y:' , y )

    line_left  = [ low [ 0 ] + x ]
    line_right = [ low [ 1 ] + y ]

    for i in range ( 2 , bins [ 0 ] ) :

        current_left  = x + line_left  [ -1 ]
        current_right = y + line_right [ -1 ]

        if current_left  < high [ 0 ] : line_left  += [ current_left  ]
        if current_right < high [ 1 ] : line_right += [ current_right ]

    grid = np.array ( [ np.round ( line_left , 3 ) ,  np.round ( line_right , 3 ) ] )
    return grid

low  = [ -1.0 , -5.0 ]
high = [  1.0 ,  5.0 ]
create_uniform_grid ( low , high ) # [test]

# -------------------------------------------------------

def discretize(sample, grid):
    """
    Parameters:
       sample : array_like
       grid : list of array    
    Returns: discretized_sample : array_like
    """

    assert ( sample.shape [ 0 ] == 2 )
    assert ( grid.shape   [ 0 ] == 2 ) and ( grid.shape [ 1 ] > 0 )

    x = 0 ; y = 0

    for i , element in enumerate ( sample ) :

        if i == 0 :
            for y_grid in range ( grid.shape [ 1 ] ) :
                if element >= grid [ 0 , y_grid ] : x = y_grid+1

        if i == 1 :
            for y_grid in range ( grid.shape [ 1 ] ) :
                if element >= grid [ 1 , y_grid ] : y = y_grid+1
    
    return [ x , y ]
    # return indexes for sample >= grid

# -------------------------------------------------------

# Test with a simple grid and some samples
grid = create_uniform_grid([-1.0, -5.0], [1.0, 5.0])
samples = np.array(
    [[-1.0 , -5.0],
     [-0.81, -4.1],
     [-0.8 , -4.0],
     [-0.5 ,  0.0],
     [ 0.2 , -1.9],
     [ 0.8 ,  4.0],
     [ 0.81,  4.1],
     [ 1.0 ,  5.0]])
discretized_samples = np.array([discretize(sample, grid) for sample in samples])
print("\nSamples:", repr(samples), sep="\n")
print("\nDiscretized samples:", repr(discretized_samples), sep="\n")

import matplotlib.collections as mc

def visualize_samples(samples, discretized_samples, grid, low=None, high=None):

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Show grid
    ax.xaxis.set_major_locator(plt.FixedLocator(grid[0]))
    ax.yaxis.set_major_locator(plt.FixedLocator(grid[1]))
    ax.grid(True)
    
    # If bounds (low, high) are specified, use them to set axis limits
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # Otherwise use first, last grid locations as low, high (for further mapping discretized samples)
        low = [splits[0] for splits in grid]
        high = [splits[-1] for splits in grid]

    print ( 'samples:' , np.round ( samples , 3 ) )
    print ( 'discretized_samples' , discretized_samples )
    print ( '' )

    # Map each discretized sample (which is really an index) to the center of corresponding grid cell
    grid_extended = np.hstack((np.array([low]).T, grid, np.array([high]).T))  # add low and high ends
    print ( 'grid_extended:' , grid_extended )
    grid_centers = (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2  # compute center of each grid cell
    print ( 'grid_centers:' , grid_centers )
    locs = np.stack(grid_centers[i, discretized_samples[:, i]] for i in range(len(grid))).T  # map discretized samples
    print ( 'locs:' , locs )
    
    print ( '' )
    print ( 'example:' )
    print ( 'original grid:' , [ grid_extended [ 0 , 3 ] , grid_extended [ 1 , 3 ] ] )
    print ( 'modded grid:'   , [ grid_centers  [ 0 , discretized_samples [ 3 , 0 ] ] ,
                                 grid_centers  [ 1 , discretized_samples [ 3 , 1 ] ] ] )
    print ( 'original simple:' , [ samples [ 3 , 0 ] , samples [ 3 , 1 ] ] )
    print ( 'modded simple:'   , [ grid_centers [ 0 , discretized_samples [ 3 , 0 ] ] ,
                                   grid_centers [ 1 , discretized_samples [ 3 , 1 ] ] ] )

    ax.plot(samples[:, 0], samples[:, 1], 'o')  # plot original samples
    ax.plot(locs[:, 0], locs[:, 1], 's')  # plot discretized samples in mapped locations
    ax.add_collection(mc.LineCollection(list(zip(samples, locs)), colors='orange'))  # add a line connecting each original-discretized sample
    ax.legend(['original', 'discretized'])

    
visualize_samples(samples, discretized_samples, grid, low, high)

# Create a grid to discretize the state space
print ( 'env.observation_space.low:' , env.observation_space.low )
print ( 'env.observation_space.high:' , env.observation_space.high )
state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))
np.round ( state_grid , 3 )

# Obtain some samples from the space, discretize them, and then visualize them
state_samples = np.array([env.observation_space.sample() for i in range(10)])
discretized_state_samples = np.array([discretize(sample, state_grid) for sample in state_samples])
visualize_samples(state_samples, discretized_state_samples, state_grid,
                  env.observation_space.low, env.observation_space.high)
plt.xlabel('position'); plt.ylabel('velocity');  # axis labels for MountainCar-v0 state space

# -------------------------------------------------------

class QLearningAgent:

    def __init__(self, env, state_grid, alpha=0.02, gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505):
        """Initialize variables, create grid for discretization."""
        # Environment info
        self.env = env
        self.state_grid = state_grid
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)  # n-dimensional state space
        self.action_size = self.env.action_space.n  # 1-dimensional discrete action space
        self.seed = np.random.seed(seed)
        print("Environment:", self.env)
        print("State space size:", self.state_size)
        print("Action space size:", self.action_size)
        
        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor

        self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
        self.epsilon_decay_rate = epsilon_decay_rate # how quickly should we decrease epsilon
        self.min_epsilon = min_epsilon
        
        # Create Q-table
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        print("Q table size:", self.q_table.shape)

    def preprocess_state(self, state):
        """Map a continuous state to its discretized representation."""

        # print ( 'initial state:' , np.round ( state , 3 ) )
        discretized_state = discretize ( state, state_grid )
        # print ( 'discretized state:' , np.round ( discretized_state , 3 ) )
        return tuple ( discretized_state ) # tuple to return as state arguments for the QTable

    def reset_episode(self, state):
        """Reset variables for a new episode."""
        # Gradually decrease exploration rate
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

        # Decide initial action
        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        return self.last_action
    
    def reset_exploration(self, epsilon=None):
        """Reset exploration rate used when training."""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def act(self, state, reward=None, done=None, mode='train'):
        """Pick next action and update internal Q table (when mode != 'test')."""
        state = self.preprocess_state(state)
        if mode == 'test':
            # Test mode: Simply produce an action
            action = np.argmax(self.q_table[state])
        else :
            # it is smart , doesn't update Q-table in testing session
            old_value = self.q_table [ self.last_state + ( self.last_action , ) ]

            self.q_table [ self.last_state + ( self.last_action , ) ] = old_value + \
                           self.alpha * ( reward + self.gamma * max ( self.q_table [ state ] ) - old_value )

            # Exploration vs. exploitation
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                # Pick a random action
                action = np.random.randint(0, self.action_size)
            else:
                # Pick the best action from Q table
                action = np.argmax(self.q_table[state])

        # Roll over current state, action for next step
        self.last_state = state
        self.last_action = action
        return action

    
q_agent = QLearningAgent(env, state_grid)

# -------------------------------------------------------

def run(agent, env, num_episodes=20000, mode='train'):
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        state = env.reset()
        action = agent.reset_episode(state)
        total_reward = 0
        done = False

        # Roll out steps until done
        while not done:
            state, reward, done, info = env.step(action)
            total_reward += reward
            action = agent.act(state, reward, done, mode)

        # Save final score
        scores.append(total_reward)
        
        # Print episode stats
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score

            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()

    return scores

# scores = run(q_agent, env)
# Plot scores obtained per episode
# plt.plot(scores)
# plt.title("Scores");

def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.plot(scores); plt.title("Scores");
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean);
    return rolling_mean

# rolling_mean = plot_scores(scores)

# Run in test mode and analyze scores obtained
test_scores = run(q_agent, env, num_episodes=100, mode='test')
print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
_ = plot_scores(test_scores, rolling_window=10)

def plot_q_table(q_table):
    """Visualize max Q-value for each state and corresponding action."""
    q_image = np.max(q_table, axis=2)       # max Q-value for each state
    q_actions = np.argmax(q_table, axis=2)  # best action for each state

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(q_image, cmap='jet');
    cbar = fig.colorbar(cax)
    for x in range(q_image.shape[0]):
        for y in range(q_image.shape[1]):
            ax.text(x, y, q_actions[x, y], color='white',
                    horizontalalignment='center', verticalalignment='center')
    ax.grid(False)
    ax.set_title("Q-table, size: {}".format(q_table.shape))
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')


plot_q_table(q_agent.q_table)

# -------------------------------------------------------
# Modify the Grid

state_grid_new = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(12, 12))
q_agent_new = QLearningAgent(env, state_grid_new)
q_agent_new.scores = []  # initialize a list to store scores for this agent

# Train it over a desired number of episodes and analyze scores
# Note: This cell can be run multiple times, and scores will get accumulated
q_agent_new.scores += run(q_agent_new, env, num_episodes=50000)  # accumulate scores
rolling_mean_new = plot_scores(q_agent_new.scores)

# Run in test mode and analyze scores obtained
test_scores = run(q_agent_new, env, num_episodes=100, mode='test')
print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
_ = plot_scores(test_scores)

# Visualize the learned Q-table
plot_q_table(q_agent_new.q_table)

# -------------------------------------------------------
# Watch a Smart Agent

state = env.reset()
score = 0
img = plt.imshow(env.render(mode='rgb_array'))
for t in range(1000):
    action = q_agent_new.act(state, mode='test')
    img.set_data(env.render(mode='rgb_array')) 
    plt.axis('off')
    display.display(plt.gcf())
    display.clear_output(wait=True)
    state, reward, done, _ = env.step(action)
    score += reward
    if done:
        print('Score: ', score)
        break
        
env.close ()
