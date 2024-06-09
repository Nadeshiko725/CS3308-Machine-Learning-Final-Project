from calendar import c
from distutils.log import Log
from re import T
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
import os
import sys
import pickle
import torch.nn.functional as F
import torch_geometric  # 用于图神经网络的PyTorch扩展库
import torch_geometric.transforms
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, global_mean_pool
import tqdm

TRAIN_DATA_PATH = './task2/source_5/train'
TEST_DATA_PATH = './task2/source_5/test'
Log_PATH = './log'
Score_PATH = './score'

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# 数据加载和预处理
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type, dropout_rate=0.5):
        super(GNN, self).__init__()
        self.dropout_rate = dropout_rate

        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            self.fc = torch.nn.Linear(out_channels, 1)
            # self.fc = torch.nn.Linear(hidden_channels, out_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, out_channels)
            self.fc = torch.nn.Linear(out_channels, 1)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
            self.fc = torch.nn.Linear(out_channels, 1)
        elif model_type == 'Cheb':
            self.conv1 = ChebConv(in_channels, hidden_channels)
            self.conv2 = ChebConv(hidden_channels, out_channels)
            self.fc = torch.nn.Linear(out_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x

def load_data():
    train_data_list = []
    for file_name in tqdm.tqdm(os.listdir(TRAIN_DATA_PATH), desc='Loading training data', colour='blue'):
        if file_name.endswith('.pkl'):
            with open(os.path.join(TRAIN_DATA_PATH, file_name), 'rb') as f:
                data_dict = pickle.load(f)
            node_type_one_hot = F.one_hot(torch.tensor(data_dict['node_type'])).float()
            x = torch.cat([node_type_one_hot, torch.tensor(data_dict['num_inverted_predecessors']).unsqueeze(dim=1)], dim=1)
            edge_index = torch.tensor([data_dict['edge_src_index'], data_dict['edge_target_index']], dtype=torch.long)
            y = torch.tensor(data_dict['score']).float()
            train_data_list.append(Data(x=x, edge_index=edge_index, y=y))


    test_data_list = []
    for file_name in tqdm.tqdm(os.listdir(TEST_DATA_PATH), desc='Loading test data', colour='blue'):
        if file_name.endswith('.pkl'):
            with open(os.path.join(TEST_DATA_PATH, file_name), 'rb') as f:
                data_dict = pickle.load(f)
            node_type_one_hot = F.one_hot(torch.tensor(data_dict['node_type'])).float()
            x = torch.cat([node_type_one_hot, torch.tensor(data_dict['num_inverted_predecessors']).unsqueeze(dim=1)], dim=1)
            edge_index = torch.tensor([data_dict['edge_src_index'], data_dict['edge_target_index']], dtype=torch.long)
            y = torch.tensor(data_dict['score']).float()
            test_data_list.append(Data(x=x, edge_index=edge_index, y=y))

    return train_data_list, test_data_list

# 构建训练集和测试集的数据加载器
def get_data_loader(data_list, batch_size):
    return DataLoader(data_list, batch_size=batch_size, shuffle=True)

# 训练模型
def train_model(model, train_loader, optimizer, loss_fn):
    model.train()
    losses = []
    for data in train_loader:
        data = data.to('cuda')
        optimizer.zero_grad()
        out = model(data)  # 前向传播
        out = out.squeeze()
        loss = loss_fn(out, data.y)  # 计算损失
        losses.append(loss.item())
        with open(os.path.join(Score_PATH, 'log_scores.txt'), 'a') as f:
            f.write(f'train: Real：{data.y.mean().item()}，Pedict：{out.mean().item()}\n')
        with open(os.path.join(Log_PATH, 'log_loss.txt'), 'a') as f:
            f.write(f'train loss: {loss.item()}\n')
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
    return np.mean(losses)

# 评估模型
def evaluate_model(model, test_loader, loss_fn, epoch):
    model.eval()
    total_loss = 0
    batch_count = 0  # 用于计算平均损失
    with torch.no_grad():
        for data in test_loader:
            data = data.to('cuda')
            out = model(data)
            out = out.squeeze()
            loss = loss_fn(out, data.y).item()
            total_loss += loss
            batch_count += 1
            for predicted, actual in zip(out.squeeze(), data.y):
                with open(os.path.join(Score_PATH, 'log_scores.txt'), 'a') as f:
                    f.write(f'Epoch: {epoch}, Predict: {predicted.item():.4f}, Real: {actual.item():.4f}\n')
            with open(os.path.join(Log_PATH, 'log_loss.txt'), 'a') as f:
                f.write(f'test loss: {loss}\n')
    return total_loss / batch_count

# 主函数
def main():
    # 加载和预处理数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data = load_data()
    batch_size = 32  # 设置批处理大小
    train_loader = get_data_loader(train_data, batch_size)
    test_loader = get_data_loader(test_data, batch_size)

    # 定义模型参数
    in_channels = 4  # 节点特征的维度, node_type和num_inverted_predecessors
    hidden_channels = 64  # 隐藏层的通道数
    out_channels = 1  # 输出层的通道数（例如，预测一个评估值）

    # 创建模型
    model_type = 'GCN'
    model = GNN(in_channels, hidden_channels, out_channels, model_type).to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()  # 均方误差损失
    #loss_fn = torch.nn.L1Loss()

    # 记录loss的列表
    train_losses = []
    test_losses = []

    # 训练模型
    for epoch in tqdm.tqdm(range(100), desc='Training', colour='green'):
        train_loss = train_model(model, train_loader, optimizer, loss_fn)
        test_loss = evaluate_model(model, test_loader, loss_fn, epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        with open(os.path.join(Log_PATH, 'log_final_1.txt'), 'a') as f:
            f.write(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}\n')

    # 可视化loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Test Loss over Epochs')
    plt.savefig(os.path.join(Log_PATH, 'loss_plot.png'))
    plt.show()
    model_path = './model/task2_5.pth'
    torch.save(model.state_dict(), model_path)

# 运行主函数
if __name__ == "__main__":
    if not os.path.exists(Log_PATH):
        os.makedirs(Log_PATH)
    if not os.path.exists(Score_PATH):
        os.makedirs(Score_PATH)
    set_seed(10)
    main()

