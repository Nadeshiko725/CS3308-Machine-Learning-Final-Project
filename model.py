from calendar import c
from distutils.log import Log
from re import T
import numpy as np
import sklearn

import torch
import os
import sys
import pickle
import torch.nn.functional as F
import torch_geometric  # 用于图神经网络的PyTorch扩展库
import torch_geometric.transforms
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, global_mean_pool
import tqdm

TRAIN_DATA_PATH = './train'
TEST_DATA_PATH = './test'
Log_PATH = './log'

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super(GNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = torch.nn.Linear(out_channels, 1)
        # self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        return x

# 数据加载和预处理
def load_data():
    train_data_list = []
    for file_name in tqdm.tqdm(os.listdir(TRAIN_DATA_PATH), desc='Loading training data', colour='blue'):
        if file_name.endswith('.pkl'):
            with open(os.path.join(TRAIN_DATA_PATH, file_name), 'rb') as f:
                data_dict = pickle.load(f)
            x = torch.tensor(np.array([data_dict['node_type'], data_dict['num_inverted_predecessors']])).t().float()
            edge_index = torch.tensor(np.array([data_dict['edge_src_index'], data_dict['edge_target_index']])).long()
            y = torch.tensor(np.array(data_dict['score'])).float()
            data = Data(x=x, edge_index=edge_index, y=y)
            train_data_list.append(data)

    test_data_list = []
    for file_name in tqdm.tqdm(os.listdir(TEST_DATA_PATH), desc='Loading test data', colour='blue'):
        if file_name.endswith('.pkl'):
            with open(os.path.join(TEST_DATA_PATH, file_name), 'rb') as f:
                data_dict = pickle.load(f)
            x = torch.tensor(np.array([data_dict['node_type'], data_dict['num_inverted_predecessors']])).t().float()
            edge_index = torch.tensor(np.array([data_dict['edge_src_index'], data_dict['edge_target_index']])).long()
            y = torch.tensor(np.array(data_dict['score'])).float()
            data = Data(x=x, edge_index=edge_index, y=y)
            test_data_list.append(data)

    return train_data_list, test_data_list

# 构建训练集和测试集的数据加载器
def get_data_loader(data_list):
    return DataLoader(data_list, shuffle=False)

# 训练模型
def train_model(model, train_loader, optimizer, loss_fn):
    model.train()
    # losses = []
    total_loss = 0
    for data in train_loader:
        data = data.to('cuda')
        optimizer.zero_grad()
        out = model(data)  # 前向传播
        loss = loss_fn(out, data.y)  # 计算损失
        # losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
    return total_loss

# 评估模型
def evaluate_model(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to('cuda')
            out = model(data)
            total_loss += loss_fn(out, data.y).item()
    return total_loss, out.item()

# 主函数
def main():
    # 加载和预处理数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data = load_data()
    train_loader = get_data_loader(train_data)
    test_loader = get_data_loader(test_data)

    # 定义模型参数
    in_channels = 2  # 节点特征的维度, node_type和num_inverted_predecessors
    hidden_channels = 64  # 隐藏层的通道数
    out_channels = 1  # 输出层的通道数（例如，预测一个评估值）
    

    # 创建模型
    model = GNN(in_channels, hidden_channels, out_channels).to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()  # 均方误差损失
    # loss_fn = torch.nn.CrossEntropyLoss()  

    scores = []
    # 训练模型
    for epoch in tqdm.tqdm(range(50), desc='Training', colour='green'):
        train_loss = train_model(model, train_loader, optimizer, loss_fn)
        test_loss, score = evaluate_model(model, test_loader, loss_fn)
        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Score: {score}')
        with open(os.path.join(Log_PATH, 'log_final.txt'), 'a') as f:
            f.write(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Score: {score}\n')


# 运行主函数
if __name__ == "__main__":
    if not os.path.exists(Log_PATH):
        os.makedirs(Log_PATH)
    set_seed(10)
    main()