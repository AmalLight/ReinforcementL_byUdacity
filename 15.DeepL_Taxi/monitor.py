from collections import deque
from collections import defaultdict

from copy import copy

import sys
import math
import numpy as np

def interact ( env , agent , window = 100 , show = True ) :

    # agent: instance of class Agent (see Agent.py for details)
    # window: number of episodes to consider when calculating average rewards

    # Returns
    #   avg_rewards: deque containing average rewards
    #   best_avg_reward: largest value in the avg_rewards deque

    avg_rewards = deque ( maxlen = agent.episodes_numbers )
    best_avg_reward = -math.inf
    Qset = defaultdict ( lambda : np.zeros ( self.nA ) )
    samp_rewards = deque ( maxlen = window )

    for i_episode in range ( 1 , agent.episodes_numbers + 1 ) :

        state = env.reset ()
        samp_reward = 0
        
        # generate
        while True :
            action = agent.select_action ( state , i_episode ) # agent move = made an action
            next_state , reward , done , _ = env.step ( action )
            agent.step ( state , action , reward , next_state , done ) # update Q_table

            samp_reward += reward
            state = next_state

            if done:
                samp_rewards.append ( samp_reward )
                break

        if ( i_episode >= 100 ) :
            avg_reward = np.mean ( samp_rewards )
            avg_rewards.append ( avg_reward )

            if avg_reward > best_avg_reward :
               best_avg_reward = avg_reward
               Qset = copy ( agent.Q )

        if not show : continue

        print ( "\rEpisode {}/{} || Best average reward {} || trust: {} ".format (
              i_episode , agent.episodes_numbers , best_avg_reward , round ( agent.trust , 2 ) ) , end = "" )

        sys.stdout.flush ()

        if best_avg_reward >= 9.7 :
            print ( '\nEnvironment solved in {} episodes.'.format ( i_episode ), end = "" )
            break

        if i_episode == agent.episodes_numbers : print ( '\n' )
    return avg_rewards, best_avg_reward , Qset
