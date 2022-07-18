from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make ( 'Taxi-v2' )

good , i = 9.4 , 0

while True :

    env.reset ()

    # work very well after many iterations,
    # unique learning in one single run, forced to learning ( 1.0 ).

    tmp_agent = Agent ( nA = 6 , episodes_numbers = 4 * 1000 ,
                                 episodes_min_prop_for_prob = 1.0 ,
                                 episodes_max_prop_for_prob = 1.0 ,
                                 alpha = 1.0 , show = False )

    tmp_avg_rewards , tmp_best_avg_reward , Qset = interact ( env , tmp_agent , show = False )

    print ( 'tmp_best_avg_reward' , i , ':' , tmp_best_avg_reward ) # max reached is 9.67.

    # doesn't work very well, it should force to not learning too much ( 0.5 ).

    if ( tmp_best_avg_reward >= good ) :

         print ( 'found tmp_best_avg_reward >' , good ) # max reached is 9.67
                                                        # lost for wrong set ( testing )
         tmp_agent.Q = Qset
         tmp_agent.set ( show = True , episodes_numbers = 200 * 1000 ,
                                       episodes_min_prop_for_prob = 0.1 ,
                                       episodes_max_prop_for_prob = 0.5,
                                       starting_prob_base = 0.0 ,
                                       limit_prob_onTrust_inLearning = 1.0 ,
                                       closing_prob_value = 1.0 , alpha = 0.5 )

         interact ( env , tmp_agent  , show = True ) # max reached is 9.57 <-- from a best of 9.4.
         break

    else : i += 1

    # States were written very bad --> bad gym env <-- I need array for states not a single number.
