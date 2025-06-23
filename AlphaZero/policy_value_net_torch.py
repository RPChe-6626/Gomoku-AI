"""
Created on Sat Dec  8 13:02:14 2018

@author: initial-h

Modified on Tue May 13 22:06:14 2025

@author: RPChe_
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# This is the policy value network model for AlphaZero.
class PolicyValueNetModel(nn.Module):
    def __init__(self, board_width, board_height, block, planes_num):
        super().__init__()
        self.planes_num = planes_num
        self.nb_block = block
        self.board_width = board_width
        self.board_height = board_height

        # Convolutional layers here all have kernel_size=1, stride=1, padding=0. Unless mentioned, you shouldn't use nn.Sequential here.
        
        # # Backbone Network
        # self.zeropad2d
        # self.conv2d_1 channel=64
        # self.resnet = resblock * self.nb_block chanel=64 # need residual link and relu. you may use nn.ModuleList

        # # Action Network
        # self.conv2d_2 channel=2
        # self.bn_1 # need a relu and permute(0, 2, 3, 1)
        # self.flatten_layer_1
        # self.dense_layer_1 # need a log softmax

        # # Evaluation Network
        # self.conv2d_3 channel=1
        # self.bn_2 = # need a relu and permute(0, 2, 3, 1)
        # self.flatten_layer_2
        # self.dense_layer_2 # need a relu
        # self.flatten_layer_3 # need a tanh
        pass

    def resblock(self, channels):
        # Convolutional layers here all have kernel_size=3, stride=1, padding=0. You may use nn.Sequential
        # nn.Conv2d
        # nn.BatchNorm2d
        # nn.ReLU
        # nn.Conv2d
        # nn.BatchNorm2d
        pass
    
    def forward(self, x):
        act, eval = None, None
        pass
        return act, eval
        # The output of the network is log probabilities for actions and a value for the state.

# This is a wrapper class for the PolicyValueNetModel.
class PolicyValueNet():
    def __init__(self, board_width, board_height, block, init_model=None, transfer_model=None, cuda=False):
        print()
        print('building network ...')
        print()

        # Save hyperparameters
        self.planes_num = 9  # feature planes
        self.nb_block = block # resnet blocks
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.board_width = board_width
        self.board_height = board_height
        # Used as macros
        # This part is set to fit the api in tensorflow version.
        # but is not used in this Pytorch version.
        # Actually, only action_fc_test and evaluation_fc2_test are used.
        self.action_fc_train = 0
        self.evaluation_fc2_train = 1
        self.action_fc_test = 2
        self.evaluation_fc2_test = 3

        # Initialize model
        self.model = PolicyValueNetModel(self.board_width, self.board_height, self.nb_block, self.planes_num)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # lr is set during training before, but Pytorch does not support this. So if you want to configure the learning rate, you have to do it here.
        self.oppo = PolicyValueNetModel(board_width, board_height, block, self.planes_num)
        self.oppo.to(self.device)

        # Load model if specified
        if init_model is not None:
            self.restore_model(init_model)
            print('model loaded!')
        elif transfer_model is not None:
            self.restore_model(transfer_model)
            print('transfer model loaded!')
        else:
            print('can not find saved model, learn from scratch !')

    # def save_numpy(self, params):
    # def load_numpy(self, params, path='tmp/model.npy'):
    
    # def print_params(self, params):
    def print_params(self):
        for name, param in self.model.named_parameters():
            print(f"name: {name}, shape: {param.data.shape}")
        for name, param in self.model.named_buffers():
            print(f"buffer name: {name}, shape: {param.data.shape}")
    
    def policy_value(self, state_batch, actin_fc, evaluation_fc):
        '''
        input: a batch of states, actin_fc, evaluation_fc
        output: a batch of action probabilities and state values
        '''
        assert actin_fc == self.action_fc_test, "policy_value: action_fc not match"
        assert evaluation_fc == self.evaluation_fc2_test, "policy_value: evaluation_fc not match"
        # As I said, actin_fc and evaluation_fc are not used in this Pytorch version.
        act_probs, value = None, None
        pass
        return act_probs, value
    
    def policy_value_fn(self, board, actin_fc, evaluation_fc):
        '''
        input: board,actin_fc,evaluation_fc
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        '''
        # the accurate policy value fn,
        # i prefer to use one that has some randomness even when test,
        # so that each game can play some different moves, all are ok here
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, self.planes_num, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state, actin_fc, evaluation_fc)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value
    
    def policy_value_fn_random(self, board, actin_fc, evaluation_fc):
        '''
        input: board,actin_fc,evaluation_fc
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        '''
        # like paper said,
        # The leaf node sL is added to a queue for neural network
        # evaluation, (di(p), v) = fθ(di(sL)),
        # where di is a dihedral reflection or rotation
        # selected uniformly at random from i in [1..8]

        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, self.planes_num, self.board_width, self.board_height))
        
        # print('current state shape',current_state.shape)

        #add dihedral reflection or rotation
        rotate_angle = np.random.randint(1, 5)
        flip = np.random.randint(0, 2)
        equi_state = np.array([np.rot90(s, rotate_angle) for s in current_state[0]])
        if flip:
            equi_state = np.array([np.fliplr(s) for s in equi_state])
        # print(equi_state.shape)

        # put equi_state to network
        act_probs, value = self.policy_value(np.array([equi_state]), actin_fc, evaluation_fc)

        # get dihedral reflection or rotation back
        equi_mcts_prob = np.flipud(act_probs[0].reshape(self.board_height, self.board_width))
        if flip:
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
        equi_mcts_prob = np.rot90(equi_mcts_prob, 4 - rotate_angle)
        act_probs = np.flipud(equi_mcts_prob).flatten()

        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        '''
        perform a training step
        state_batch: NCHW, mcts_probs: NHW, winner_batch: N
        '''
        loss, entropy = None, None
        # value loss use MSELoss
        # policy loss use CrossEntropyLoss
        # all other parameters except bias terms should have a weight decay of 1e-4, that means 1e-4 / 2.0 * ||w||^2
        # the final loss is their sum
        pass
        return loss, entropy

    def save_model(self, model_path):
        '''
        save model to model_path
        '''
        torch.save(self.model.state_dict(), model_path)

    def restore_model(self, model_path):
        '''
        restore model from model_path
        '''
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))