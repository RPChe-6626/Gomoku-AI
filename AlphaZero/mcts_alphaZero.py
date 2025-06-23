# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 22:05:17 2018

@author: initial
"""


import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    '''
    A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    '''

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p # its the prior probability that action's taken to get this node

    pass


class MCTS(object):
    '''
    An implementation of Monte Carlo Tree Search.
    '''
    # action_fc and evaluation_fc are not used in Pytorch version, just placeholders. see details in policy_value_net_pytorch.py
    def __init__(self, policy_value_fn,action_fc,evaluation_fc, is_selfplay,c_puct=5, n_playout=400):
        '''
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        '''
        self._root = TreeNode(None, 1.0)
        # root node do not have parent ,and sure with prior probability 1

        self._policy_value_fn = policy_value_fn
        self._action_fc = action_fc
        self._evaluation_fc = evaluation_fc

        self._c_puct = c_puct
        # it's 5 in paper and don't change here,but maybe a better number exists in gomoku domain
        self._n_playout = n_playout # times of tree search
        self._is_selfplay = is_selfplay

    pass
    
    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
    '''
    AI player based on MCTS
    '''
    def __init__(self, policy_value_function,action_fc,evaluation_fc,c_puct=5, n_playout=400, is_selfplay=0):
        '''
        init some parameters
        '''
        self._is_selfplay = is_selfplay
        self.policy_value_function = policy_value_function
        self.action_fc = action_fc
        self.evaluation_fc = evaluation_fc
        # self.first_n_moves = 12
        self.first_n_moves = 4
        # For the first n moves of each game, the temperature is set to τ = 1,
        # For the remainder of the game, an infinitesimal temperature is used, τ→ 0.
        # in paper n=30, here i choose 12 for 11x11, entirely by feel
        self.mcts = MCTS(policy_value_fn = policy_value_function,
                         action_fc = action_fc,
                         evaluation_fc = evaluation_fc,
                         is_selfplay = self._is_selfplay,
                         c_puct = c_puct,
                         n_playout = n_playout)

    def set_player_ind(self, p):
        '''
        set player index
        '''
        self.player = p

    def reset_player(self):
        '''
        reset player
        '''
        self.mcts.update_with_move(-1)

    def get_action(self,board,is_selfplay,print_probs_value):
        '''
        get an action by mcts
        do not discard all the tree and retain the useful part
        '''
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move, move_probs = None, np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            
            pass

            if print_probs_value and move_probs is not None:
                act_probs, value = self.policy_value_function(board,self.action_fc,self.evaluation_fc)
                print('-' * 10)
                print('value',value)
                # print the probability of each move
                probs = np.array(move_probs).reshape((board.width, board.height)).round(3)[::-1, :]
                for p in probs:
                    for x in p:
                        print("{0:6}".format(x), end='')
                    print('\r')

            return move,move_probs

        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "Alpha {}".format(self.player)


