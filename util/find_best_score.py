import os
import pickle
from tqdm import tqdm
# Define the directory containing the .pkl files
directory = './task1/project_data/'

# Dictionary to store the highest score input for each circuit
circuit_scores = {}

# List all files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

# Iterate over all files with a progress bar
for filename in tqdm(files, desc="Processing files"):
    filepath = os.path.join(directory, filename)
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
        circuit_name = filename.split('_')[0]
        last_input = data['input'][-1]
        last_target = data['target'][-1]
        if circuit_name not in circuit_scores or last_target > circuit_scores[circuit_name][1]:
            circuit_scores[circuit_name] = (last_input, last_target)

# Print the results
for circuit, (best_input, best_score) in circuit_scores.items():
    print(f"Circuit: {circuit}, Best Input: {best_input}, Best Score: {best_score}")
