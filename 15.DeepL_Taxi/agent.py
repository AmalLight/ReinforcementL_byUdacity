import sys
import numpy as np
from collections import defaultdict

class Agent:
    
    # -----------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------

    """ prob is probability ; prop is proportion """

    """ step for init QT  = episodes_min_prop_for_prob """
    """ step for training = episodes_max_prop_for_prob """

    """ training forced to learning = starting_prob_base            """
    """ training forced to learning = limit_prob_onTrust_inLearning """

    """ step for testing = closing_prob_value """
    
    def set ( self , show = True , episodes_numbers = 200 * 1000 ,
                                   episodes_min_prop_for_prob = 0.10 ,
                                   episodes_max_prop_for_prob = 0.70 ,

                                   starting_prob_base = 0.0 ,
                                   limit_prob_onTrust_inLearning = 1.0 ,

                                   closing_prob_value = 1.0 , alpha = 1 ) :
        self.show  = show
        self.alpha = alpha

        self.episodes_numbers   = episodes_numbers
        self.starting_prob_base = starting_prob_base

        self.episodes_min_prop_for_prob = episodes_min_prop_for_prob # %
        self.episodes_max_prop_for_prob = episodes_max_prop_for_prob # %

        self.closing_prob_value = closing_prob_value

        self.episodes_min_for_prob = ( self.episodes_numbers * self.episodes_min_prop_for_prob ) # value == no %
        self.episodes_max_for_prob = ( self.episodes_numbers * self.episodes_max_prop_for_prob ) # value == no %

        self.episodes_used_to_starting = self.episodes_max_for_prob - self.episodes_min_for_prob
        self.episodes_used_after_min   = self.episodes_numbers      - self.episodes_min_for_prob

        self.episodes_used_prop_to_closing  = round ( 1 - self.episodes_max_prop_for_prob , 2 )
        self.episodes_used_to_closing = self.episodes_numbers - self.episodes_max_for_prob

        self.lr = 0
        if self.episodes_used_after_min > 0 :
           self.lr = ( 1 - self.starting_prob_base ) / self.episodes_used_to_starting

        self.limit_prob_onTrust_inLearning = limit_prob_onTrust_inLearning
        
        # -------------------------------------------------------------
        # -------------------------------------------------------------

        assert self.episodes_min_for_prob     <= self.episodes_numbers
        assert self.episodes_max_for_prob     <= self.episodes_numbers
        assert self.episodes_used_to_starting <= self.episodes_numbers
        assert self.episodes_used_after_min   <= self.episodes_numbers
        
        assert self.episodes_used_to_starting <= self.episodes_used_after_min 

        if self.lr > 0 : assert round ( self.lr * self.episodes_used_to_starting + self.starting_prob_base ) == 1

        assert round ( ( 1 - self.episodes_max_prop_for_prob ) *  self.episodes_numbers + \
                             self.episodes_max_for_prob      ) == self.episodes_numbers

        assert round ( ( 1 - self.episodes_min_prop_for_prob ) *  self.episodes_numbers + \
                             self.episodes_min_for_prob      ) == self.episodes_numbers

        assert round ( self.episodes_used_prop_to_closing * self.episodes_numbers ) == \
                       self.episodes_used_to_closing

        # -------------------------------------------------------------
        # -------------------------------------------------------------

        if self.show :

           print ( '' )
           print ( 'numbers of actions:' , self.nA               )
           print ( 'numbers episodes:'   , self.episodes_numbers )

           print ( 'starting probability:'  , self.starting_prob_base )
           print ( 'last used probability:' , self.closing_prob_value )

           print ( '' )

           print ( 'episodes min % for probability:' , self.episodes_min_prop_for_prob )
           print ( 'episodes max % for probability:' , self.episodes_max_prop_for_prob )

           print ( 'episodes min for probability:' , self.episodes_min_for_prob )
           print ( 'episodes max for probability:' , self.episodes_max_for_prob )

           print ( '' )
     
           print ( 'episodes used to starting:' , self.episodes_used_to_starting )
           print ( 'episodes used after min:'   , self.episodes_used_after_min   )

           print ( 'episodes used % to close:' , self.episodes_used_prop_to_closing )
           print ( 'episodes used to close:' , self.episodes_used_to_closing  )

           print ( '' )

           print ( 'learning rate:' , self.lr )
           print ( 'alpha value:' , self.alpha )
           print ( 'limit % on learning:' , self.limit_prob_onTrust_inLearning )
           print ( '' )

    # -----------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------

    def __init__( self , nA = 6 , episodes_numbers = 200 * 1000 ,
                                  episodes_min_prop_for_prob = 0.10 ,
                                  episodes_max_prop_for_prob = 0.70 ,

                                  starting_prob_base = 0.0 ,
                                  limit_prob_onTrust_inLearning = 1.0 ,

                                  closing_prob_value = 1.0 ,
                                  alpha = 1 , show = True ) :
        self.nA = nA
        self.Q = defaultdict ( lambda : np.zeros ( self.nA ) )
        self.set ( show , episodes_numbers ,
                          episodes_min_prop_for_prob ,
                          episodes_max_prop_for_prob ,
                          starting_prob_base ,
                          limit_prob_onTrust_inLearning ,
                          closing_prob_value , alpha )

    # -----------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------

    def select_action ( self , state , i_episode ) : # random action from 6 actions

        prob = 1

        if ( i_episode  > self.episodes_min_for_prob ) and \
           ( i_episode <= self.episodes_max_for_prob ) :

           difference = ( i_episode  -  self.episodes_min_for_prob )
           prob = self.starting_prob_base + ( self.lr * difference )
           assert prob >= self.starting_prob_base

           if prob > self.limit_prob_onTrust_inLearning :
              prob = self.limit_prob_onTrust_inLearning

        elif ( i_episode > self.episodes_max_for_prob ) : prob = self.closing_prob_value
        self.trust = prob ;

        choice = np.random.choice ( np.arange ( 2 ) , p=[1-prob,prob] )

        if choice == 1 : return np.argmax ( self.Q [ state ] )
        else           : return np.random.choice ( self.nA )

        # AThere are 6 discrete deterministic actions:
        # - 0: move south , 1: move north , 2: move east , 3: move west
        # - 4: pickup passenger
        # - 5: drop off passenger

    # -----------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------

    def step ( self , state , action , reward , next_state , done , gamma=1 ) :

        old_Qvalue = self.Q [ state ][ action ]
        
        # --------------------------------------
        # --------------------------------------

        # new_state_action = self.Q [ next_state ] [ action ] # Sarsa 0

        new_state_action = np.max ( self.Q [ next_state ] ) # Q-Learning
        assert np.max ( self.Q [ next_state ] ) == self.Q [ next_state ] [ np.argmax ( self.Q [ next_state ] ) ]

        # probs = np.array ( [ 1 / self.nA ] * self.nA )              # Sarsa Expected
        # new_state_action = np.dot ( self.Q [ next_state ] , probs ) # Sarsa Expected
        # i can't choose this new_state_action bucause i don't know the right prob..,
        #
        # the numbers of states were written very bad --> bad gym env.

        # --------------------------------------
        # --------------------------------------

        sarsa = old_Qvalue + self.alpha * ( reward + ( gamma * new_state_action ) - old_Qvalue )

        # Sarsa = punish only new training
        # sarsa = old_Qvalue + self.alpha * ( reward + ( gamma * new_state_action ) - old_Qvalue )

        # TD is too much difficult to manage
        # TD = punish old and new training
        # sarsa = ( ( 1 - self.alpha ) * old_Qvalue ) + self.alpha * ( reward + ( gamma * new_state_action ) )

        self.Q [ state ] [ action ] = sarsa
