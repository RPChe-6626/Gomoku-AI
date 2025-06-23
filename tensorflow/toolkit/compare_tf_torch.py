from policy_value_net_tensorlayer import PolicyValueNet as tfv
from policy_value_net_torch import PolicyValueNet as torchv
import numpy as np
import torch

tfn = tfv(11, 11, 19, init_model='./model_11_11_5/best_policy.model', cuda=False)
torchn = torchv(11, 11, 19, init_model='./torch/converted_model.pt', cuda=False)

# input = np.random.rand(1, 9, 11, 11).astype(np.float32)
input = np.zeros((1, 9, 11, 11), dtype=np.float32)
# randomly set some positions to 1
input[0, 0, 5, 5] = 1
input[0, 1, 5, 6] = 1
input[0, 2, 5, 4] = 1
input[0, 3, 6, 5] = 1
input[0, 4, 4, 5] = 1
input[0, 5, 5, 7] = 1
input[0, 6, 5, 3] = 1
input[0, 7, 6, 6] = 1
input[0, 8, 4, 4] = 1

act1, val1 = tfn.policy_value(input, tfn.action_fc_test, tfn.evaluation_fc2_test)
act2, val2 = torchn.policy_value(input, torchn.action_fc_test, torchn.evaluation_fc2_test)

print(act1)
print(act1.shape, val1)
print(act2)
print(act2.shape, val2)