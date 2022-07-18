import sys
import gym
import numpy as np

from collections import defaultdict, deque
from plot_utils import plot_values

import matplotlib.pyplot as plt
import check_test

env = gym.make('CliffWalking-v0')

print(env.action_space)
print(env.observation_space)

# --------------------------------------------------

# define the optimal state-value function
V_opt = np.zeros((4,12))
V_opt[0:13][0] = -np.arange(3, 15)[::-1]
V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
V_opt[3][0] = -13

plot_values(V_opt)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def AI_policy_Sarsa1 ( Q , state , action , next_state , alpha , reward , info , gamma=1 ) :

    new_state_action = Q [ next_state ][ action ]
    old_Qvalue       = Q [      state ][ action ]

    sarsa = alpha * ( reward + (gamma * new_state_action) - old_Qvalue )
    Q [ state ] [ action ] = ( old_Qvalue + sarsa )
    return Q

def generate_episode ( env , Q , i , alpha , gamma=1 , AI_policy=AI_policy_Sarsa1 ) :
    # print ( 'Started to play game:' , i )

    states = 48
    episode = []
    state = env.reset ()

    while True:

        action = np.argmax ( Q [ state ] ) # best policy, no need for epsilon
        next_state , reward , done , info = env.step ( action )
        Q = AI_policy ( Q , state , action , next_state , alpha , reward , info )

        episode.append ( ( state , action , reward ) )
        state = next_state

        if done or reward == -100 : break

    # print ( 'Ended to play game:' , i )
    return episode , Q

def sarsa ( env , num_episodes , alpha , gamma=1.0 ) :
    Q = defaultdict ( lambda: np.zeros ( env.action_space.n ) )
    
    states  = 48
    for state in range ( states ) :
        for action in range ( env.action_space.n ) :
            Q [ state ][ action ] = 0

    # initialize performance monitor
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()   
        
        episode , Q = generate_episode ( env , Q , i_episode , alpha , gamma )

    print ( '' )

    # print Q 3x..
    for i , key in enumerate ( Q ) :
        print ( 'Q' , i , '=' , key , ':' , Q [ key ] )
        print ( '' )
        if i == 3-1 : break
    
    return Q

# --------------------------------------------------

# obtain the estimated optimal policy and corresponding action-value function
Q_sarsa = sarsa(env, 5000, .01)

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
plot_values(V_sarsa)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def AI_policy_Sarsa2 ( Q , state , action , next_state , alpha , reward , info , gamma=1 ) :

    new_state_action = Q [ next_state ][ np.argmax ( Q [ next_state ] ) ]
    old_Qvalue       = Q [      state ][ action ]

    sarsa = alpha * ( reward + (gamma * new_state_action) - old_Qvalue )
    Q [ state ] [ action ] = ( old_Qvalue + sarsa )
    return Q

def q_learning ( env , num_episodes , alpha , gamma=1.0 ) :
    Q = defaultdict ( lambda: np.zeros ( env.action_space.n ) )
    
    states  = 48
    for state in range ( states ) :
        for action in range ( env.action_space.n ) :
            Q [ state ][ action ] = 0

    # initialize performance monitor
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()   
        
        episode , Q = generate_episode ( env , Q , i_episode , alpha , gamma , AI_policy_Sarsa2 )

    print ( '' )

    # print Q 3x..
    for i , key in enumerate ( Q ) :
        print ( 'Q' , i , '=' , key , ':' , Q [ key ] )
        print ( '' )
        if i == 3-1 : break
    
    return Q

# ----------------------------------------------------------------------------

# obtain the estimated optimal policy and corresponding action-value function
Q_sarsamax = q_learning(env, 5000, .01)

# print the estimated optimal policy
policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
check_test.run_check('td_control_check', policy_sarsamax)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsamax)

# plot the estimated optimal state-value function
plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def AI_policy_Sarsa3 ( Q , state , action , next_state , alpha , reward , info , gamma=1 ) :

    old_Qvalue = Q [ state ][ action ]

    # if reward == -100 : None
    # if reward == -100 : next_state = 36
    # if reward == -100 : probs = [ 0 ] * 4

    probs = [ info [ 'prob' ] / 4 ] * 4 # info [ 'prob' ] = 100% , the same for all
    assert probs == [ 0.25 , 0.25 , 0.25 , 0.25 ]
    probs = np.array ( probs )

    new_state_action = np.dot ( Q [ next_state ] , probs ) # dot "has" sum inside
    sarsa = alpha * ( reward + ( gamma * new_state_action ) - old_Qvalue )
    Q [ state ] [ action ] = ( old_Qvalue + sarsa )

    return  Q

def expected_sarsa ( env , num_episodes , alpha , gamma=1.0 ) :
    Q = defaultdict ( lambda: np.zeros ( env.action_space.n ) )
    
    states  = 48
    for state in range ( states ) :
        for action in range ( env.action_space.n ) :
            Q [ state ][ action ] = 0

    # initialize performance monitor
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()   
        
        episode , Q = generate_episode ( env , Q , i_episode , alpha , gamma , AI_policy_Sarsa3 )

    print ( '' )

    # print Q 3x..
    for i , key in enumerate ( Q ) :
        print ( 'Q' , i , '=' , key , ':' , Q [ key ] )
        print ( '' )
        if i == 3-1 : break
    
    return Q

# ----------------------------------------------------------------------------

# obtain the estimated optimal policy and corresponding action-value function
Q_expsarsa = expected_sarsa(env, 5000, 0.01)

# print the estimated optimal policy
policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_expsarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_expsarsa)

# plot the estimated optimal state-value function
plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])
