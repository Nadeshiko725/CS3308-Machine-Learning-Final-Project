import os
import sys
import numpy as np
import pickle

TRAIN_DATA_PATH = './task2/source_5/train'
TEST_DATA_PATH = './task2/source_5/test'
DATA_PATH = './task2/source_5/src/'

def divide_dataset(train_ratio=0.8):
    # 把DATA_PATH目录下的数据集分为训练集和测试集，分别存储在TRAIN_DATA_PATH和TEST_DATA_PATH目录下
    if not os.path.exists(TRAIN_DATA_PATH):
        os.makedirs(TRAIN_DATA_PATH)
    if not os.path.exists(TEST_DATA_PATH):
        os.makedirs(TEST_DATA_PATH)
    
    for file_name in os.listdir(DATA_PATH):
        # 将全部的数据集分为训练集和测试集，分割比例为train_ratio
        if file_name.endswith('.pkl'):
            with open(os.path.join(DATA_PATH, file_name), 'rb') as f:
                data_dict = pickle.load(f)
            if np.random.rand() < train_ratio:
                save_path = os.path.join(TRAIN_DATA_PATH, file_name)
            else:
                save_path = os.path.join(TEST_DATA_PATH, file_name)
            with open(save_path, 'wb') as f:
                pickle.dump(data_dict, f)

if __name__ == "__main__":
    divide_dataset()
    print("Dataset divided successfully!")