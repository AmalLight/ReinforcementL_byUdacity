import sys
import gym
import numpy as np

from collections import defaultdict
from plot_utils  import plot_blackjack_values, plot_policy

env = gym.make('Blackjack-v0')

print(env.observation_space)
print(env.action_space)

for i_episode in range(3):
    state = env.reset()
    while True:
        print(state)
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            print('End game! Reward: ', reward)
            print('You won :)\n') if reward > 0 else print('You lost :(\n')
            break

def generate_episode_from_limit_stochastic(bj_env):
    episode = []
    state = bj_env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

for i in range(3):
    print(generate_episode_from_limit_stochastic(env))

# ------------------------------------------------------------
# ------------------------------------------------------------

def mc_prediction_q ( env , num_episodes , generate_episode , gamma = 1.0 ) :

    # print env.action_space.n
    print ( 'env.action_space.n:'  , env.action_space.n )
    print ( 'np.zeros:' , np.zeros ( env.action_space.n ) )

    # initialize empty dictionaries of arrays
    R = defaultdict ( lambda: np.zeros ( env.action_space.n ) ) # template for free dict -> { .. : {} }
    N = defaultdict ( lambda: np.zeros ( env.action_space.n ) ) # template for free dict -> { .. : {} }
    Q = defaultdict ( lambda: np.zeros ( env.action_space.n ) ) # template for free dict -> { .. : {} }

    # print value for first N
    for key in N.keys () :
        print ( 'first N_value:' , N [ key ] ) # empty
        break
    
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # create episodes
        episode = generate_episode ( env )
        # generate_episode is generate_episode_from_limit_stochastic

        for i , step1 in enumerate ( episode ) :
            state  =         step1 [ 0 ]
            action =         step1 [ 1 ]
            reward = sum ( [ step2 [ 2 ] for step2 in episode [ i : ] ] ) * gamma
            
            # gamma from github uses gamma**i for i in i:

            # fc  = my_first_card   = state1 [ 0 ]
            # sc  = the_second_card = state1 [ 1 ]
            # ace = i_can_use_ace   = state1 [ 2 ]
            # unhashable type: 'numpy.ndarray' ; unhashable type: 'list'

            R [ state ] [ action ] += reward
            N [ state ] [ action ] += 1
            Q [ state ] [ action ]  = R [ state ] [ action ] / N [ state ] [ action ]
    return  Q

# ------------------------------------------------------------
# ------------------------------------------------------------

# obtain the action-value function
Q = mc_prediction_q ( env , 500000 , generate_episode_from_limit_stochastic )

# obtain the corresponding state-value function
V_to_plot = dict (

    ( k , ( k [ 0 ] >  18 ) * ( np.dot ( [ 0.8 , 0.2 ] , v ) ) +
          ( k [ 0 ] <= 18 ) * ( np.dot ( [ 0.2 , 0.8 ] , v ) ) ) \

    for k , v in Q.items ()
)

# plot the state-value function
plot_blackjack_values(V_to_plot)

# ------------------------------------------------------------
# ------------------------------------------------------------

def generate_episode_from_limit_stochastic ( bj_env , Q=None , more=0.8 , less=0.2 ) :
    episode = []
    state = bj_env.reset()
    an = env.action_space.n

    while True:
        if Q and state in Q :

            Q_choice = np.argmax ( Q [ state ] ) # best policy
            probs = [ more , less ] if ( Q_choice == 0 ) else [ less , more ]

        action = np.random.choice ( np.arange ( an ) , p = ( probs if Q and state in Q else None ) )
        next_state , reward , done , info = bj_env.step ( action )
        episode.append ( ( state , action , reward ) )
        state = next_state

        if done: break
    return episode

def mc_control ( env , num_episodes=500000 , alpha=0.2 , gamma=1.0 , more=0.8 , less=0.2 ) :

    # initialize empty dictionary of arrays
    R = defaultdict ( lambda: np.zeros ( env.action_space.n ) ) # template for free dict -> { .. : {} }
    N = defaultdict ( lambda: np.zeros ( env.action_space.n ) ) # template for free dict -> { .. : {} }
    Q = defaultdict ( lambda: np.zeros ( env.action_space.n ) ) # template for free dict -> { .. : {} }

    lr = less / num_episodes # you can create Matrix "More and Less" based on dictionary for each state
    done = 0                 # this matrix would have an incremental "more" value for any hit it was received

    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            
        # create episodes
        episode = generate_episode_from_limit_stochastic ( env , Q , more+lr*i_episode , less-lr*i_episode )
        done = i_episode

        for i , step1 in enumerate ( episode ) :
            state  =         step1 [ 0 ]
            action =         step1 [ 1 ]
            reward = sum ( [ step2 [ 2 ] for step2 in episode [ i : ] ] ) * gamma

            R [ state ] [ action ] += reward
            N [ state ] [ action ] += 1
            Q [ state ] [ action ]  = R [ state ] [ action ] / N [ state ] [ action ]

            Qv = Q [ state ] [ action ]

            loss = reward - Qv # how many rewards can i have in future? convenience
            final_convenience = alpha * loss

            Q [ state ] [ action ] = Qv + final_convenience

    print ( '' )
    
    # print final more and less
    print ( 'done:'       ,         done )
    print ( 'final lr:'   ,      lr*done )
    print ( 'final more:' , more+lr*done )
    print ( 'final less:' , less-lr*done )
    
    print ( '' )

    # print Q 3x..
    for i , key in enumerate ( Q ) :
        print ( 'Q' , i , '=' , key , ':' , Q [ key ] )
        print ( '' )
        if i == 3-1 : break
    
    print ( '' )

    # €-greedy
    policy = {}
    # for state in Q.keys () :
    #    state_values = []
    #    for value in Q [ state ].keys () : state_values += [ Q [ state ][ value ] ]
    #    policy [ state ] = state_values.index ( max ( state_values ) )

    policy = { k : np.argmax ( v ) for k , v in Q.items () }

    # print policy 3x..
    for i , key in enumerate ( policy ) :
        print ( 'policy' , i , '=' , key , ':' , policy [ key ] )
        print ( '' )
        if i == 3-1 : break
    
    return policy, Q # mine is simpler but also less accurate

# obtain the estimated optimal policy and action-value function
policy, Q = mc_control ( env , num_episodes = 500000 , alpha = 0.2 , more = 0.80 , less = 0.20 )

# obtain the corresponding state-value function
V = dict ( ( k , np.max ( v ) ) for k , v in Q.items () )

# plot the state-value function
plot_blackjack_values ( V )

# plot the policy
plot_policy ( policy )
