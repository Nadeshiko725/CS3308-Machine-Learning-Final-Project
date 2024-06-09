import sys
import numpy as np
import torch
import os
import pickle
import torch.nn.functional as F
import torch_geometric 
import torch_geometric.transforms
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, global_mean_pool, TransformerConv
import re
import abc_py as abcPy
import numpy as np
import pickle
import queue

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

def aig_process(circuitName,actions):
    prevState = '../InitialAIG/train/' + circuitName + '.aig'
    libFile = '../lib/7nm/7nm.lib'
    logFile = './123.log'
    if(actions==''):
        nextState = './origin.aig'
    else:
        nextState = './' + actions + '.aig'
    synthesisOpToPosDic = {
        0: "refactor",
        1: "refactor -z",
        2: "rewrite",
        3: "rewrite -z",
        4: "resub",
        5: "resub -z",
        6: "balance"
    }
    action_cmd = ""
    for action in actions:
        action_cmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = f"../yosys/yosys-abc -c \"read {prevState}; {action_cmd} read_lib {libFile}; write {nextState}; print_stats\" > {logFile}"
    os.system(abcRunCmd)
    _abc = abcPy.AbcInterface()
    _abc.start()
    _abc.read(nextState)
    data = {}
    numNodes = _abc.numNodes()
    data['node_type'] = np.zeros(numNodes, dtype=int)
    data['num_inverted_predecessors'] = np.zeros(numNodes, dtype=int)
    edge_src_index = []
    edge_target_index = []
    for nodeIdx in range(numNodes):
        aigNode = _abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        data['num_inverted_predecessors'][nodeIdx] = 0
        if nodeType == 0 or nodeType == 2:
            data['node_type'][nodeIdx] = 0
        elif nodeType == 1:
            data['node_type'][nodeIdx] = 1
        else:
            data['node_type'][nodeIdx] = 2
            if nodeType == 4:
                data['num_inverted_predecessors'][nodeIdx] = 1
            if nodeType == 5:
                data['num_inverted_predecessors'][nodeIdx] = 2
        if (aigNode.hasFanin0()):
            fanin = aigNode.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
        if (aigNode.hasFanin1()):
            fanin = aigNode.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)

    data['edge_src_index'] = edge_src_index
    data['edge_target_index'] = edge_target_index
    return data

class Node:
    def __init__(self,name, start_status='',_iter=0): # ��ʼ���м�ڵ�Ĳ���
        self.data = None
        self.state = start_status
        self.name=name
        self.father = None
        self.iter=_iter
        self.transform()
        with torch.no_grad():
            self.g = model1(self.data).item()
            self.h = model2(self.data).item()
        self.f=self.g+self.h

    def transform(self):
        data=aig_process(self.name,self.state)
        node_type_one_hot = F.one_hot(torch.tensor(data['node_type'])).float()
        x = torch.cat([node_type_one_hot, torch.tensor(data['num_inverted_predecessors']).unsqueeze(dim=1)], dim=1)
        edge_index = torch.tensor(np.array([data['edge_src_index'], data['edge_target_index']])).long()
        self.data=Data(x=x,edge_index=edge_index)

    def __lt__(self,other):
        return self.f > other.f

class AStar:
    def __init__(self, max_iter):
        self.path = []
        self.open_list = queue.PriorityQueue()
        self.max_iter=max_iter

    def select_current(self) -> Node:
        return self.open_list.get()

    def explore_neighbors(self, current_node):
        directions = range(0,7)
        for direction in directions:
            new_state=current_node.state+str(direction)
            new_node=Node(current_node.name,new_state,_iter=current_node.iter+1)
            new_node.father = current_node
            self.open_list.put(new_node)

    def find(self,name,start_status):
        start_node = Node(name,start_status,0)
        self.open_list.put(start_node)
        prev_node=None
        current_node=None
        while True:
            current_node = self.select_current()
            if prev_node != None:
                if abs(prev_node.f-current_node.f)<0.001:
                    self.open_list=queue.PriorityQueue()
            print(current_node.state)
            print(current_node.g)
            print(current_node.h)
            if current_node is None:
                return None
            self.explore_neighbors(current_node)
            prev_node=current_node
            if current_node.iter==self.max_iter:
                break
        while current_node.father is not None:
            self.path.insert(0, current_node.state)
            current_node = current_node.father
        return self.path
    
    def clear(self):
        self.open_list = queue.PriorityQueue()
        self.path = []

if __name__=='__main__':
    name_list=['adder','alu2','apex3','arbiter','b2','c1355','ctrl','frg1','i7','int2float','log2','m3','max512','multiplier']
    in_channels = 4  
    hidden_channels = 64 
    out_channels = 1 
    model_type = 'GCN'
    model1 = GNN(in_channels, hidden_channels, out_channels, model_type)
    model2 = GNN(in_channels, hidden_channels, out_channels, model_type)
    model1.load_state_dict(torch.load('./task1.pth',map_location=torch.device('cpu')))
    model2.load_state_dict(torch.load('./task2_5.pth',map_location=torch.device('cpu')))
    model1.eval()
    model2.eval()
    solution=AStar(max_iter=10)
    ans=[]
    for _name in name_list:
        ans.append(solution.find(name=_name,start_status=''))
        solution.clear()
        print('--'*20)
    for item in ans:
        print(item)
        print('--'*20)
