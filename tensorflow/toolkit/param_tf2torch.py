import tensorflow as tf
import numpy as np
import torch
from collections import OrderedDict
import re

# 模型保存的目录
checkpoint_path = "./model_11_11_5/best_policy.model"

# 创建检查点读取器
reader = tf.train.NewCheckpointReader(checkpoint_path)

# 获取所有变量名和形状
var_to_shape_map = reader.get_variable_to_shape_map()

def tf_to_torch_name(tf_name):
    """将 TensorFlow 参数名转换为 PyTorch 参数名"""
    
    # 处理 ResNet 块参数
    resnet_pattern = r'model/resnet_(conv2d|bn)_(\d+)_(\d+)/(kernel|bias|gamma|beta|moving_mean|moving_variance)'
    match = re.match(resnet_pattern, tf_name)
    if match:
        layer_type, block_num, sub_block_num, param_type = match.groups()
        # 转换为基于0的索引
        sub_block_num = int(sub_block_num) - 1
        
        # 映射参数名称
        if layer_type == 'conv2d':
            # Conv2d 参数
            param_map = {
                'kernel': 'weight',
                'bias': 'bias'
            }
            index = sub_block_num * 3  # 每个子块中第一个卷积的位置
            return f"resnet.{block_num}.{index}.{param_map[param_type]}"
        else:  # layer_type == 'bn'
            # BatchNorm2d 参数
            param_map = {
                'gamma': 'weight',
                'beta': 'bias',
                'moving_mean': 'running_mean',
                'moving_variance': 'running_var'
            }
            index = sub_block_num * 3 + 1  # 每个子块中第一个BN的位置
            return f"resnet.{block_num}.{index}.{param_map[param_type]}"
    
    # 处理非 ResNet 参数
    non_resnet_pattern = r'model/(conv2d|bn|dense_layer|flatten_layer)_(\d+)/(kernel|bias|gamma|beta|moving_mean|moving_variance|W|b)'
    match = re.match(non_resnet_pattern, tf_name)
    if match:
        layer_type, layer_num, param_type = match.groups()
        
        # 映射参数名称
        if layer_type in ['conv2d', 'dense_layer', 'flatten_layer']:
            param_map = {
                'kernel': 'weight',
                'bias': 'bias',
                'W': 'weight',
                'b': 'bias'
            }
            return f"{layer_type}_{layer_num}.{param_map[param_type]}"
        else:  # layer_type == 'bn'
            param_map = {
                'gamma': 'weight',
                'beta': 'bias',
                'moving_mean': 'running_mean',
                'moving_variance': 'running_var'
            }
            return f"bn_{layer_num}.{param_map[param_type]}"
    
    # 如果没有匹配到任何模式，返回原始名称
    return tf_name

params = OrderedDict()

for name in var_to_shape_map:
    if 'Adam' in name or 'model' not in name:
        continue
    torch_name = tf_to_torch_name(name)
    weight = reader.get_tensor(name)
    if len(weight.shape) == 4:
        weight = np.transpose(weight, (3, 2, 0, 1))
    elif len(weight.shape) == 2:
        weight = np.transpose(weight)
    tensor = torch.from_numpy(weight)
    params[torch_name] = tensor
    print(torch_name, tensor.shape)

assert len(params) == 248, f"Expected 248 parameters, but got {len(params)}"

torch.save(params, './torch/converted_model.pt')
