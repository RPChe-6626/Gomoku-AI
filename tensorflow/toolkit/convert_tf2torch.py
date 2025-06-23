import os
import numpy as np
import tensorflow as tf
import torch
from policy_value_net_tensorlayer import PolicyValueNet as TFPolicyValueNet
from policy_value_net_torch import PolicyValueNet as PTPolicyValueNet

def convert_tf_to_pytorch(board_width=11, board_height=11, block=5):
    # 创建输出目录
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    
    # 步骤1: 加载TensorFlow模型并保存为NumPy格式
    print("正在加载TensorFlow模型...")
    tf_model = TFPolicyValueNet(board_width, board_height, block, 
                              transfer_model='model_11_11_5/best_policy.model')
    
    # 获取所有网络参数
    tf_params = tf_model.network_params
    # 保存为NumPy格式
    tf_model.save_numpy(tf_params)  # 注意这里传递了参数
    print("TensorFlow模型已保存为NumPy格式")
    
    # 步骤2: 创建PyTorch模型并从NumPy加载参数
    print("正在创建PyTorch模型...")
    pt_model = PTPolicyValueNet(board_width, board_height, block)
    
    # 从NumPy加载参数
    pt_model.load_numpy('tmp/model.npy')
    
    # 保存为PyTorch格式
    pt_model.save_model('tmp/best_policy_pytorch.model')
    print("PyTorch模型已保存")
    
    return pt_model

if __name__ == "__main__":
    # 根据你的原始模型尺寸设置参数
    convert_tf_to_pytorch(board_width=11, board_height=11, block=5)