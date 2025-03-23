import json
import pickle

import numpy as np


def convert_pickle_to_json(pickle_file, json_file):
    # 加载pickle文件
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    # 转换numpy数组为列表
    converted_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            # 确保数组是2维的
            if value.ndim == 1:
                value = value.reshape(-1, 1)
            converted_data[key] = {
                "shape": list(value.shape),  # 确保shape是列表
                "data": value.astype(
                    float
                ).tolist(),  # 确保数据是原生Python浮点数
            }

    # 保存为JSON，使用compact格式
    with open(json_file, "w") as f:
        json.dump(converted_data, f, allow_nan=True)

    # 打印数据结构以验证
    print("Converted data structure:")
    for key, value in converted_data.items():
        print(f"{key}: shape={value['shape']}")


# 转换文件
convert_pickle_to_json("sample_weight.pkl", "sample_weight.json")
