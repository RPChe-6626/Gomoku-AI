def filter_non_adam_variables(variables_list):
    """
    过滤掉包含'Adam'的变量名
    :param variables_list: 变量名列表，格式为["变量名: name, 形状: shape", ...]
    :return: 不包含'Adam'的变量列表
    """
    filtered_vars = []
    for var in variables_list:
        # 分割变量名部分
        var_name_part = var.split(",")[0]
        var_name = var_name_part.split(":")[1].strip()
        
        # 检查是否包含'Adam'
        if "resnet" in var_name:
            filtered_vars.append(var)
    return filtered_vars

# 示例使用
if __name__ == "__main__":
    # 这里应该是你的完整变量列表，由于太长，我只展示如何使用
    # 假设你的变量列表存储在variables.txt文件中
    with open("./filtered_params.txt", "r") as f:
        variables = f.readlines()
    
    # 过滤变量
    filtered = filter_non_adam_variables(variables)
    
    # 输出结果
    # for var in filtered:
    #     print(var.strip())
    
    # 或者保存到新文件
    with open("resnet_params.txt", "w") as f:
        f.writelines(filtered)