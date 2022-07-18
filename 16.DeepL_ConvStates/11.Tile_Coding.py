
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)

# Create an environment
env = gym.make('Acrobot-v1')
env.seed(505);

# Explore state (observation) space
print("State space:", env.observation_space)
print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)

# Explore action space
print("Action space:", env.action_space)

# ---------------------------------------------------------

def create_tiling_grid ( low , high , bins = ( 10 , 10 ) , offsets = ( 0.0 , 0.0 ) ) :
    """
    Parameters:
        ...
        offsets : tuple
    Returns:
        grid : list of array
    """
    
    assert bins [ 0 ] == bins [ 1 ]

    x =  low [ 0 ] - high [ 0 ]
    if x < 0 : x *= -1
    x /=  bins [ 0 ]
    x = round ( x , 3 )
    print ( 'x:' , x )
    print ( 'x offset:' , offsets [ 0 ] )

    y = low [ 1 ] - high [ 1 ]
    if y < 0 : y *= -1
    y /=  bins [ 1 ]
    y = round ( y , 3 )
    print ( 'y:' , y )
    print ( 'y offset:' , offsets [ 1 ] )

    line_left  = [ low [ 0 ] + offsets [ 0 ] + x ]
    line_right = [ low [ 1 ] + offsets [ 1 ] + y ]

    for i in range ( 2 , bins [ 0 ] ) :

        current_left  = ( x + line_left  [ -1 ] )
        current_right = ( y + line_right [ -1 ] )

        if current_left  < high [ 0 ] : line_left  += [ current_left  ]
        if current_right < high [ 1 ] : line_right += [ current_right ]

    grid = np.array ( [ np.round ( line_left , 3 ) ,  np.round ( line_right , 3 ) ] )
    return grid

low  = [ -1.0 , -5.0 ]
high = [  1.0 ,  5.0 ]
create_tiling_grid ( low , high , bins = ( 10 , 10 ) , offsets = ( -0.1 , 0.5 ) ) # [test]

# ---------------------------------------------------------

def create_tilings ( low , high , tiling_specs ) :
    """
    Parameters:
        tiling_specs : list of tuples
    Returns:
        tilings : A list of tilings
    """
    tilings = []
    for tuple_tiling in tiling_specs :
        bins    = tuple_tiling [ 0 ]
        offsets = tuple_tiling [ 1 ]
        tilings += [ create_tiling_grid ( low , high , bins , offsets ) ]

    # print ( 'tilings:' , tilings )
    return  tilings

# Tiling specs: [(<bins>, <offsets>), ...]
tiling_specs = [ ( ( 10 , 10 ) , ( -0.066 , -0.33 ) ) ,
                 ( ( 10 , 10 ) , ( 0.0    ,  0.0  ) ) ,
                 ( ( 10 , 10 ) , ( 0.066  ,  0.33 ) ) ]

tilings = create_tilings(low, high, tiling_specs)


# ---------------------------------------------------------

from matplotlib.lines import Line2D

def visualize_tilings(tilings):
    """Plot each tiling as a grid."""
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    linestyles = ['-', '--', ':']
    legend_lines = []

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, grid in enumerate(tilings):
        for x in grid[0]:
            l = ax.axvline(x=x, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], label=i)
        for y in grid[1]:
            l = ax.axhline(y=y, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
        legend_lines.append(l)
    ax.grid('off')
    ax.legend(legend_lines, ["Tiling #{}".format(t) for t in range(len(legend_lines))], facecolor='white', framealpha=0.9)
    ax.set_title("Tilings")
    return ax  # return Axis object to draw on later, if needed


visualize_tilings(tilings);

# ---------------------------------------------------------

def discretize ( sample , grid ) :
    assert ( len ( sample ) == 2 )
    assert ( grid.shape [ 0 ] == 2 ) and ( grid.shape [ 1 ] > 0 )
    x = 0 ; y = 0

    for i , element in enumerate ( sample ) :

        if i == 0 :
            for y_grid in range ( grid.shape [ 1 ] ) :
                if element >= grid [ 0 , y_grid ] : x = y_grid+1

        if i == 1 :
            for y_grid in range ( grid.shape [ 1 ] ) :
                if element >= grid [ 1 , y_grid ] : y = y_grid+1
    
    return [ x , y ]


def tile_encode ( sample , tilings , flatten = False ) :
    """
    Parameters:
       flatten : bool
       If true, flatten the resulting binary arrays into a single long vector.
    Returns:
       encoded_sample : discretized list or array
    """
    assert ( len ( sample ) == 2 )

    encoded_sample = []
    for tiling in tilings :
        discretize_tiling = discretize ( sample , tiling )
        
        # print ( 'tiling:' , tiling )
        # print ( 'sample:' , sample )

        if not flatten :
               encoded_sample += [ ( discretize_tiling [ 0 ] , 
                                     discretize_tiling [ 1 ] ) ]
        else :
               encoded_sample += [ discretize_tiling [ 0 ] , 
                                   discretize_tiling [ 1 ] ]
    return encoded_sample

# Test with some sample values
samples = [ (-1.2 , -5.1 ) ,
            (-0.75,  3.25) ,
            (-0.5 ,  0.0 ) ,
            ( 0.25, -1.9 ) ,
            ( 0.15, -1.75) ,
            ( 0.75,  2.5 ) ,
            ( 0.7 , -3.7 ) ,
            ( 1.0 ,  5.0 ) ]

encoded_samples  =  [  tile_encode ( sample , tilings ) for sample in samples ]
print ( "\nSamples:",         repr (          samples ) , sep = "\n" )
print ( "\nEncoded samples:", repr (  encoded_samples ) , sep = "\n" )

# ---------------------------------------------------------

from matplotlib.patches import Rectangle

def visualize_encoded_samples(samples, encoded_samples, tilings, low=None, high=None):
    """Visualize samples by activating the respective tiles."""
    samples = np.array(samples)  # for ease of indexing

    # Show tiling grids
    ax = visualize_tilings(tilings)
    
    # If bounds (low, high) are specified, use them to set axis limits
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # Pre-render (invisible) samples to automatically set reasonable axis limits, and use them as (low, high)
        ax.plot(samples[:, 0], samples[:, 1], 'o', alpha=0.0)
        low = [ax.get_xlim()[0], ax.get_ylim()[0]]
        high = [ax.get_xlim()[1], ax.get_ylim()[1]]

    # Map each encoded sample (which is really a list of indices) to the corresponding tiles it belongs to
    tilings_extended = [np.hstack((np.array([low]).T, grid, np.array([high]).T)) for grid in tilings]  # add low and high ends
    tile_centers = [(grid_extended[:, 1:] + grid_extended[:, :-1]) / 2 for grid_extended in tilings_extended]  # compute center of each tile
    tile_toplefts = [grid_extended[:, :-1] for grid_extended in tilings_extended]  # compute topleft of each tile
    tile_bottomrights = [grid_extended[:, 1:] for grid_extended in tilings_extended]  # compute bottomright of each tile

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for sample, encoded_sample in zip(samples, encoded_samples):
        for i, tile in enumerate(encoded_sample):
            # Shade the entire tile with a rectangle
            topleft = tile_toplefts[i][0][tile[0]], tile_toplefts[i][1][tile[1]]
            bottomright = tile_bottomrights[i][0][tile[0]], tile_bottomrights[i][1][tile[1]]
            ax.add_patch(Rectangle(topleft, bottomright[0] - topleft[0], bottomright[1] - topleft[1],
                                   color=colors[i], alpha=0.33))

            # In case sample is outside tile bounds, it may not have been highlighted properly
            if any(sample < topleft) or any(sample > bottomright):
                # So plot a point in the center of the tile and draw a connecting line
                cx, cy = tile_centers[i][0][tile[0]], tile_centers[i][1][tile[1]]
                ax.add_line(Line2D([sample[0], cx], [sample[1], cy], color=colors[i]))
                ax.plot(cx, cy, 's', color=colors[i])
    
    # Finally, plot original samples
    ax.plot(samples[:, 0], samples[:, 1], 'o', color='r')

    ax.margins(x=0, y=0)  # remove unnecessary margins
    ax.set_title("Tile-encoded samples")
    return ax

visualize_encoded_samples(samples, encoded_samples, tilings);

# ---------------------------------------------------------

class QTable :
    def __init__(self, state_size, action_size):
        """
        Parameters:
            state_size  : tuple/discrete values
            action_size : int, Number of discrete actions
        """
        print ( 'state_size:' , state_size )
        print ( 'action_size:' , action_size )

        self.state_size  = state_size
        self.action_size = action_size

        self.q_table = np.zeros ( shape = state_size + (action_size,) )
        print("QTable(): size =", self.q_table.shape)

class TiledQTable:    
    def __init__ ( self , low , high , tiling_specs , action_size ) :
        """
        Parameters:
            tiling_specs : list of tuples, A sequence of (bins, offsets)
            action_size  : int , Number of discrete actions in action space.
        """
        self.tilings = create_tilings ( low , high , tiling_specs )
        self.state_sizes = [ tuple ( len ( splits ) + 1 for splits in  tiling_grid )                                                                    for tiling_grid in self.tilings ]
        self.action_size = action_size
        self.q_tables = [ QTable ( state_size , self.action_size ) for state_size in self.state_sizes ]

        print("TiledQTable(): no. of internal tables = ", len ( self.q_tables ) )
    
    def get ( self , state , action ) :
        """
        Parameters:
            state : array
            action : int
        Returns:
            Q-value of given <state, action> pair, averaged from all internal Q-tables.
        """
        encoded_states = tile_encode ( state , self.tilings )
        
        value_average = 0

        for i in range ( len ( self.q_tables ) ) :
            q_table  =  self.q_tables [ i ].q_table
            value_average += q_table  [ encoded_states [ i ][ 0 ] ,
                                        encoded_states [ i ][ 1 ] , action ]

        return ( value_average / len ( self.q_tables ) )

    def update ( self , state , action , updated_value , alpha = 0.1):
        """
        Parameters:
            state  : array
            action : int
            value  : float
            alpha  : float
        """
        encoded_states = tile_encode ( state , self.tilings )

        for i in range ( len ( self.q_tables ) ) :
            encoded_x = encoded_states [ i ][ 0 ]
            encoded_y = encoded_states [ i ][ 1 ]
            encoded_z = action

            old_value = \
            self.q_tables [ i ].q_table [ encoded_x , encoded_y , encoded_z ]
            self.q_tables [ i ].q_table [ encoded_x , encoded_y , encoded_z ] = \
                                          alpha * updated_value + (1.0 - alpha) * old_value
# Test with a sample Q-table
tq = TiledQTable(low, high, tiling_specs, 2)
s1 = 3; s2 = 4; a = 0; q = 1.0
print("[GET]    Q({}, {}) = {}".format(samples[s1], a, tq.get(samples[s1], a)))  # check value at sample = s1, action = a
print("[UPDATE] Q({}, {}) = {}".format(samples[s2], a, q)); tq.update(samples[s2], a, q)  # update value for sample with some common tile(s)
print("[GET]    Q({}, {}) = {}".format(samples[s2], a, tq.get(samples[s2], a)))  # check value at sample = s1, action = a
print("[GET]    Q({}, {}) = {}".format(samples[s1], a, tq.get(samples[s1], a)))  # check value again, should be slightly updated

# ---------------------------------------------------------

def create_tiling_grid ( low , high , bins = ( 10 , 10 ) , offsets = ( 0.0 , 0.0 ) ) :

    print ( 'low:' , low )
    print ( 'high:' , high )
    print ( 'bins:' , bins )
    print ( 'offsets:' , offsets )
    
    last_bins = bins [ 0 ]
    for el in bins : assert last_bins == el
    
    to_add = [ ( ( low [ i ] - high [ i ] ) / last_bins ) for i in range ( len ( low ) ) ]

    for i in range ( len ( to_add ) ) :
        if  to_add [i]  < 0 :
            to_add [i] *= -1
    
    last_elements = [ l for l in low ]
    for i in range ( len ( last_elements ) ) : last_elements [i] += offsets [i]
    for i in range ( len ( last_elements ) ) : last_elements [i] += to_add  [i]

    list_for_elements = [ [l] for l in last_elements  ]
    
    for i1 , el in enumerate ( last_elements ) :
        for i2 in range ( 2 , bins [ 0 ] ) :
            
            current_element = ( list_for_elements [i1] [-1] + to_add [i1] )

            if  current_element < high [ i1 ] :
                list_for_elements [ i1 ] += [ current_element ]

    tiling_grid = np.array ( list_for_elements )
    print ( 'create_tiling_grid:' , tiling_grid )
    return tiling_grid

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

def discretize ( sample , grid ) :
    discretize_array = [ 0 ] * len ( sample )

    for i1 , sample_element in enumerate ( sample ) :
        for i2 , grid_element in enumerate ( grid [ i1 , : ] ) :
            if sample_element >= grid_element : discretize_array [ i1 ] = i2 +1
    
    return discretize_array

def tile_encode ( sample , tilings , flatten = False ) :
    encoded_sample = []
    for tiling in tilings :
        discretize_tiling = discretize ( sample , tiling )

        if flatten : encoded_sample +=           discretize_tiling
        else       : encoded_sample += [ tuple ( discretize_tiling ) ]
            
    return encoded_sample

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

class QLearningAgent ():

    def __init__ ( self , env , low , high , tiling_specs , action_size ,
                   alpha = 0.02 , gamma = 0.99 , seed = 505 ,
                   epsilon = 1.0 , epsilon_decay_rate = 0.9995 , min_epsilon = .01 ) :

        self.tilings = create_tilings ( low , high , tiling_specs )
        self.state_sizes = [ tuple ( len ( splits ) + 1 for splits in  tiling_grid ) \
                                                                   for tiling_grid in self.tilings ]
        self.action_size = action_size
        self.q_tables = [ QTable ( state_size , self.action_size ) for state_size in self.state_sizes ]

        print("TiledQTable(): no. of internal tables = ", len ( self.q_tables ) )

        self.env = env
        self.seed = np.random.seed ( seed )

        print ( "Environment:"         , self.env         )
        print ( "State  space sizes:"  , self.state_sizes )
        print ( "Action space  size:"  , self.action_size )
        
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        
        self.last_state  = [ 0 ] * 5
        self.last_action =   0

        self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
        self.epsilon_decay_rate = epsilon_decay_rate # how quickly should we decrease epsilon
        self.min_epsilon = min_epsilon

    def get ( self , state , action ) :
        encoded_states = tile_encode ( state , self.tilings )

        value_average = 0
        for i in range ( len ( self.q_tables ) ) :
            q_table  =  self.q_tables [ i ].q_table
            value_average += q_table  [ tuple ( tuple ( encoded_states [ i ] ) + (action,) ) ]

        return ( value_average / len ( self.q_tables ) )

    def update ( self , state , action , update_value , alpha = 0.1 ) :
        encoded_states = tile_encode ( state , self.tilings )

        for i in range ( len ( self.q_tables ) ) :
            old_value = \
            self.q_tables [ i ].q_table [ tuple ( tuple ( encoded_states [ i ] ) + (action,) ) ]
            self.q_tables [ i ].q_table [ tuple ( tuple ( encoded_states [ i ] ) + (action,) ) ] = \
                                          alpha * update_value + (1.0 - alpha) * old_value

    # -----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------

    def reset_episode ( self , state ) :
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon  = max ( self.epsilon , self.min_epsilon )

        get_values = [ 0 ] * self.action_size

        for i in range ( self.action_size ) :
            get_values [ i ] = self.get ( state , i )

        return np.argmax ( get_values )

    # state is next_state
    def act ( self , state , reward = None , done = None , mode = 'train' ) :

        best_action = None
        if mode == 'test': best_action = self.reset_episode ( state )
        else :

            best_action = self.reset_episode ( state ,             )
            best_value  = self.get           ( state , best_action )
            update_value = reward + ( self.gamma * best_value )

            self.update ( self.last_state , self.last_action , update_value )

            do_exploration = np.random.uniform ( 0 , 1 ) < self.epsilon
            if do_exploration : best_action = np.random.randint ( 0 , self.action_size )

        self.last_state  = state
        self.last_action = best_action
        return best_action

n_bins = 5
bins = tuple ( [ n_bins ] * env.observation_space.shape [ 0 ] )
offset_pos = (              env.observation_space.high  - \
                            env.observation_space.low ) / ( 3 * n_bins )

tiling_specs = [ ( bins , -offset_pos ) ,
                 ( bins , tuple ( [ 0.0 ] * env.observation_space.shape [ 0 ] ) ) ,
                 ( bins , offset_pos ) ]

q_agent = QLearningAgent ( env , env.observation_space.low  ,
                                 env.observation_space.high , tiling_specs , 
                                 env.action_space.n )

# ---------------------------------------------------------

def run ( agent , env , num_episodes = 10000 , mode = 'train' ) :

    scores = []
    max_avg_score = -np.inf

    print ( 'starting' )

    for i_episode in range ( 1 , num_episodes+1 ) :
        state = env.reset ()
        agent.last_state = state

        action = agent.reset_episode ( state )
        agent.last_action = action

        total_reward = 0
        done = False

        while not done:
            state, reward, done, info = env.step ( action )
            total_reward += reward
            action = agent.act ( state , reward , done , mode )

        scores.append ( total_reward )
        
        if  len ( scores ) > 100 :
            avg_score = np.mean ( scores [ -100 : ] )

            if avg_score > max_avg_score : max_avg_score = avg_score

        if  i_episode % 100 == 0:
            print ( "\rEpisode {}/{} | Max Average Score: {}".format ( 
                    i_episode , num_episodes , max_avg_score ) , end = "" )

            sys.stdout.flush ()
    return scores

scores = run ( q_agent , env )
scores = run ( q_agent , env , mode = 'test' )
scores
