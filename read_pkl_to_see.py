import numpy as np
import sklearn
import os
import torch
import sys
import pickle

def read_pkl_to_see(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    pkl_path = '../send_1615/adder_4.pkl'
    data = read_pkl_to_see(pkl_path)
    print(data)
