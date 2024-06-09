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

TRAIN_DATA_PATH = './train'
TEST_DATA_PATH = './test'
Log_PATH = './log'
Score_PATH = './score'

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type, dropout_rate=0.5):
        super(GNN, self).__init__()
        self.dropout_rate = dropout_rate

        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            self.fc = torch.nn.Linear(out_channels, 1)
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

def get_data_loader(data_list, batch_size):
    return DataLoader(data_list, batch_size=batch_size, shuffle=True)

def train_model(model, train_loader, optimizer, loss_fn):
    model.train()
    losses = []
    for data in train_loader:
        data = data.to('cuda')
        optimizer.zero_grad()
        out = model(data)
        out = out.squeeze()
        loss = loss_fn(out, data.y)
        losses.append(loss.item())
        with open(os.path.join(Score_PATH, 'log_scores.txt'), 'a') as f:
            f.write(f'train: Real：{data.y.mean().item()}，Predict：{out.mean().item()}\n')
        with open(os.path.join(Log_PATH, 'log_loss.txt'), 'a') as f:
            f.write(f'train loss: {loss.item()}\n')
        loss.backward()
        optimizer.step()
    return np.mean(losses)

def evaluate_model(model, test_loader, loss_fn, epoch):
    model.eval()
    total_loss = 0
    batch_count = 0
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

def run_experiment(model_type, train_loader, test_loader, in_channels, hidden_channels, out_channels, device, epochs=20):
    model = GNN(in_channels, hidden_channels, out_channels, model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    train_losses = []
    test_losses = []
    for epoch in tqdm.tqdm(range(epochs), desc=f'Training {model_type}', colour='green'):
        train_loss = train_model(model, train_loader, optimizer, loss_fn)
        test_loss = evaluate_model(model, test_loader, loss_fn, epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f'Epoch: {epoch:03d}, Model: {model_type}, Train loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    return train_losses, test_losses

def main():
    if not os.path.exists(Log_PATH):
        os.makedirs(Log_PATH)
    if not os.path.exists(Score_PATH):
        os.makedirs(Score_PATH)
    set_seed(10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data = load_data()
    batch_size = 32
    train_loader = get_data_loader(train_data, batch_size)
    test_loader = get_data_loader(test_data, batch_size)

    in_channels = 4
    hidden_channels = 64
    out_channels = 1

    models = ['GCN', 'GAT', 'SAGE']
    colors = ['blue', 'orange', 'green']
    all_train_losses = []
    all_test_losses = []

    for model_type, color in zip(models, colors):
        train_losses, test_losses = run_experiment(model_type, train_loader, test_loader, in_channels, hidden_channels, out_channels, device)
        all_train_losses.append((train_losses, color, model_type))
        all_test_losses.append((test_losses, color, model_type))

    plt.figure(figsize=(12, 8))
    for losses, color, label in all_train_losses:
        plt.plot(losses, label=f'{label} Train Loss', color=color, linestyle='--')
    for losses, color, label in all_test_losses:
        plt.plot(losses, label=f'{label} Test Loss', color=color)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Test Loss over Epochs for GCN, GAT, and SAGE')
    plt.savefig(os.path.join(Log_PATH, 'combined_loss_plot.png'))
    plt.show()

if __name__ == "__main__":
    main()
