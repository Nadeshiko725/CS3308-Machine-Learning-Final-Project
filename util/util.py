import os
import re
import abc_py as abcPy
import numpy as np
import pickle
import random 
from tqdm import tqdm


def aig_process(prev_state, state, score):
    dataset_dir = './task1/dataset/'
    circuitName, actions = state.split('_')
    if prev_state:
        # print(f'origin {actions}')
        prevState = prev_state
        prev_actions = prev_state.split('_')[1]
        prev_actions = prev_actions.split('.')[0]
        # print(f'prev {prev_actions}')

        new_actions = actions[len(prev_actions):]
        actions = new_actions
        # print(f'update {actions}')
    else:
        # print('new')
        prevState = './InitialAIG/train/' + circuitName + '.aig'
    
    libFile = './lib/7nm/7nm.lib'
    logFile = dataset_dir + 'log/' + state + '.log'
    nextState = dataset_dir + 'AIG/' + state + '.aig'
    nextBench = dataset_dir + 'benchmarks/' + state + '.bench'
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
    abcRunCmd = f"./yosys/yosys-abc -c \"read {prevState}; {action_cmd} read_lib {libFile}; write {nextState}; print_stats\" > {logFile}"
    os.system(abcRunCmd)
    

    # extract featurs
    _abc = abcPy.AbcInterface()
    _abc.start()
    _abc.read(nextState)
    data = {}
    numNodes = _abc.numNodes()
    data['node_type']  = np.zeros(numNodes, dtype=int)
    data['num_inverted_predecessors'] = np.zeros(numNodes, dtype = int)
    edge_src_index = []
    edge_target_index = []
    for nodeIdx in range (numNodes):
        aigNode = _abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        data['num_inverted_predecessors'][nodeIdx] = 0
        if nodeType == 0 or nodeType ==2:
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
    data['score'] = score
    # sprint(f'current score {data["score"]}')
    save_dir = dataset_dir +'test/'+ state + '.pkl'
    with open(save_dir, 'wb') as f:
        pickle.dump(data, f)
    return nextState
    
def process_all_pkls(pkl_path):
    pkl_files = [os.path.join(pkl_path, file) for file in os.listdir(pkl_path) if file.endswith('.pkl')]
    
    print("Processing PKL files...")
    for pkl_file in tqdm(pkl_files, desc="PKL files"):
    #for pkl_file in pkl_files: 
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            inputs = data['input']
            targets = data['target']
            
            prev_state = None
            for state, score in tqdm(zip(inputs, targets), desc="Inputs", leave=False, total=len(inputs)):
                prev_state = aig_process(prev_state, state, score)
            
def process_all_missing_pkls(missing_files_path):
    with open(missing_files_path, 'r') as f:
        pkl_files = [line.strip() for line in f]
    
    print("Processing missing PKL files...")
    for pkl_file in tqdm(pkl_files, desc="PKL files"):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            inputs = data['input']
            targets = data['target']
            
            prev_state = None
            for state, score in tqdm(zip(inputs, targets), desc="Inputs", leave=False, total=len(inputs)):
                prev_state = aig_process(prev_state, state, score)

def process_all_pkls_test(pkl_path):
    pkl_files = [os.path.join(pkl_path, file) for file in os.listdir(pkl_path) if file.endswith('.pkl')]
    
    print("Processing Test PKL files...")
    for pkl_file in tqdm(pkl_files, desc="PKL files"):
    #for pkl_file in pkl_files: 
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            inputs = data['input']
            targets = data['target']
            final_inputs = inputs[-1]
            final_targets = targets[-1]
            prev_state = None
            aig_process(prev_state, final_inputs, final_targets)


def split_data(pkl_path, train_ratio=0.6):
    # 创建train和test文件夹
    train_dir = os.path.join(pkl_path, 'train')
    test_dir = os.path.join(pkl_path, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 获取所有的.pkl文件
    pkl_files = [os.path.join(pkl_path, file) for file in os.listdir(pkl_path) if file.endswith('.pkl')]

    # 随机打乱文件列表
    random.shuffle(pkl_files)

    # 计算train和test的分界点
    train_size = int(len(pkl_files) * train_ratio)

    # 分割文件并移动
    train_files = pkl_files[:train_size]
    test_files = pkl_files[train_size:]

    print("Moving files to train directory...")
    for file in tqdm(train_files, desc="Training files"):
        os.rename(file, os.path.join(train_dir, os.path.basename(file)))

    print("Moving files to test directory...")
    for file in tqdm(test_files, desc="Testing files"):
        os.rename(file, os.path.join(test_dir, os.path.basename(file)))

    print(f"Total files: {len(pkl_files)}")
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")

def begin_list():
    file_path = './task1/project_data/train/'
    find_path = './task1/dataset/PKL/'
    missing_files = []
    pkl_files = [os.path.join(file_path, file) for file in os.listdir(file_path) if file.endswith('.pkl')]
    for pkl_file in tqdm(pkl_files, desc="PKL files"):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            inputs = data['input']
            last_input = inputs[-1]
            search_target = find_path + last_input + '.pkl'
            if os.path.exists(search_target):
                print('exists')
            else:
                missing_files.append(pkl_file)
    with open('./missing_files.txt', 'w') as f:
        for files in missing_files:
            f.write(f"{files}\n")

def copy_file(src, dst):
    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:
            fdst.write(fsrc.read())

def select_train():
    file_path = './task1/project_data/train/'
    find_path = './task1/dataset/PKL/'
    target_path = './task1/dataset/train/'
    pkl_files = [os.path.join(file_path, file) for file in os.listdir(file_path) if file.endswith('.pkl')]
    for pkl_file in tqdm(pkl_files, desc="PKL files"):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            inputs = data['input']
            last_input = inputs[-1]
            search_target = find_path + last_input + '.pkl'
            if os.path.exists(search_target):
                target_file = os.path.join(target_path, os.path.basename(search_target))
                copy_file(search_target, target_file)
            else:
                print('No exists!')
                print(target_file)
                print(search_target)
                print(pkl_file)

def task2_check():
    file_path = './task2/project_data2/'
    find_path = './task1/dataset/PKL/'
    missing_steps =[]
    pkl_files = [os.path.join(file_path, file) for file in os.listdir(file_path) if file.endswith('.pkl')]
    for pkl_file in tqdm(pkl_files, desc="PKL files"):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            inputs = data['input']
            for input in inputs:
                search_dir = find_path + input + '.pkl'
                if not os.path.exists(search_dir):
                    name = pkl_file +'|' + input
                    missing_steps.append(name)
    with open('./missing_task2.txt', 'w') as f:
        for files in missing_steps:
            f.write(f"{files}\n")                

def task2_train_select(directory, seed=42, percentage=0.5):
    random.seed(seed)
    files = [file for file in os.listdir(directory) if file.endswith('.pkl')]
    t2_src_directory = './task2/project_data2/' 
    selected_files = random.sample(files, int(len(files) * percentage))
    log_file = './selected_train.log'
    with open(log_file, 'w') as log:
        for file in selected_files:
            log.write(file + '\n')
    
    pkl_files = [os.path.join(directory, file) for file in selected_files]
    save_index = 0
    for pkl_file in tqdm(pkl_files, desc="PKL files"):
        with open(pkl_file, 'rb') as f:  # pkl_file = adder01
            data = pickle.load(f)
            inputs = data['input']
            targets = data['target']
            modified_src = os.path.join(t2_src_directory, os.path.basename(pkl_file))
            with open(modified_src, 'rb') as f2:
                data_2 = pickle.load(f2)
                inputs_2 = data_2['input']
                targets_2 = data_2['target']   
                
                for i, input in enumerate(inputs):
                    find_path = './task1/dataset/PKL/'
                    search_dir = find_path + input + '.pkl'
                    if not os.path.exists(search_dir):
                        print(f"ERROR: File {search_dir} not found")
                    else:
                        with open(search_dir, 'rb') as src_2:
                            data_src = pickle.load(src_2)
                            if i < len(targets_2):
                                data_src['score'] = targets_2[i]
                            else:
                                print(f"ERROR: No corresponding target for input {input}")

                            # 保存修改后的data_src
                            save_path = os.path.join('./task2/train/', f"{save_index}.pkl")
                            with open(save_path, 'wb') as f_save:
                                pickle.dump(data_src, f_save)
                            save_index += 1


def eval_yosys(state):
    circuitName, actions = state.split('_')
    circuitPath = './InitialAIG/train/' + circuitName + '.aig'
    libFile = './lib/7nm/7nm.lib'
    logFile = 'alu2.log'
    nextState = state + '.aig'  # current AIG file
    benchmark = state + '.aig'
    synthesisOpToPosDic = {
        0: 'refactor',
        1: 'refactor -z',
        2: 'rewrite',
        3: 'rewrite -z',
        4: 'resub',
        5: 'resub -z',
        6: 'balance'
    }
    actionCmd = ''
    for action in actions:
        actionCmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = "./yosys/yosys-abc -c \"read " + circuitPath + "; " + actionCmd + "; " \
                "read_lib " + libFile + "; write " + nextState + "; print_stats\" > " + logFile
    os.system(abcRunCmd)


    # First part
    abcRunCmd = "./yosys/yosys-abc -c \"read " + nextState + "; read_lib " + libFile + "; map; topo; stime\" > " + logFile
    os.system(abcRunCmd)
    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
    eval = float(areaInformation[-9]) * float(areaInformation[-4])

    # Second part
    RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
    abcRunCmd = "./yosys/yosys-abc -c \"read " + circuitPath + "; " + RESYN2_CMD + " read_lib " + libFile + "; write " + nextState + "; write_bench -l " + benchmark + "; map; topo; stime\" > " + logFile
    os.system(abcRunCmd)
    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
    baseline = float(areaInformation[-9]) * float(areaInformation[-4])
    eval = 1 - eval / baseline

    print(eval)
    return eval

def evaluate_circuits(states):
    results = {}
    for state in states:
        results[state] = eval_yosys(state)
    return results



if __name__=="__main__":
    src_data_dir = './task1/project_data/'
    # for file in os.listdir(src_data_dir):
    #     name_id = file[:-4]
    #     print (name_id)
    #     aig_process(name_id)
    #     break
    # with open('./task1/project_data/adder_0.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # print(data)
    # split_data('./task1/project_data')
    #aig_process()
    
    #process_all_pkls('./task1/project_data/train/')
    # begin_list()
    #process_all_missing_pkls('./missing_files.txt')
    #process_all_pkls_test('./task1/project_data/test/')
    # select_train()
    #task2_train_select('./task1/project_data/train/',42,0.5)
    # List of circuit states to evaluate
circuit_states = [
    'adder_1000000000',
    'alu2_1000000000',
    'apex3_1100000000',
    'arbiter_0000000000',
    'b2_1000000000',
    'c1355_4444444444',
    'ctrl_6666666666',
    'frg1_1000000000',
    'i7_1200000000',
    'int2float_1100000000',
    'log2_6262622222',
    'm3_1331000000',
    'max512_1100000000',
    'multiplier_0600000000'
]

# Evaluate the circuits
results = evaluate_circuits(circuit_states)

# Print results
for state, eval in results.items():
    print(f"Circuit {state}: Evaluation Score = {eval}")