import pickle
import sys
import argparse
import numpy as np
import pandas as pd
from os.path import join, dirname, abspath, expanduser
import matplotlib.pyplot as plt

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATA_DIR = join(ROOT_DIR, 'data', 'experiments')
sys.path.append(ROOT_DIR)

from lgp.geometry.trajectory import compute_path_length

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python process_data.py data.p')
parser.add_argument('--name', help='The scenario name of the domain, problem file', type=str, default='ground_truth.p')
args = parser.parse_args()
data_file = join(DATA_DIR, args.name)

with open(data_file, 'rb') as f:
    data = pickle.load(f)

total = len(data)
print('Total data: ', total)
single_success, dynamic_success = 0, 0
single_reduction_ratio, dynamic_reduction_ratio = [], []
single_symbolic_plan_time, dynamic_symbolic_plan_time = [], []
dynamic_path_reduction = []
dynamic_solve_nlp = {}
dynamic_geometric_plan_time = {}
for segment in data:
    print(segment)
    segment = data[segment]
    single_success += segment['single_success']
    if segment['single_success']:
        single_reduction_ratio.append(segment['single_reduction_ratio'])
        single_symbolic_plan_time.append(segment['single_symbolic_plan_time'])
        single_path = compute_path_length(segment['single_actual_path'])
    dynamic_success += segment['dynamic_success']
    if segment['dynamic_success']:
        dynamic_reduction_ratio.append(segment['dynamic_reduction_ratio'])
        dynamic_symbolic_plan_time.extend(segment['dynamic_symbolic_plan_time'].values())
        dynamic_path = compute_path_length(segment['dynamic_actual_path'])
    if segment['dynamic_success'] and segment['single_success']:
        dynamic_path_reduction.append(dynamic_path / single_path)
    for t in segment['dynamic_num_failed_plans']:
        length = len(segment['dynamic_plans'][t][0])
        if length not in dynamic_solve_nlp:
            dynamic_solve_nlp[length] = []
        dynamic_solve_nlp[length].append(segment['dynamic_num_failed_plans'][t] + 1)
    for t in segment['dynamic_geometric_plan_time']:
        if t in segment['dynamic_plans']:
            length = len(segment['dynamic_plans'][t][0])
            if length not in dynamic_geometric_plan_time:
                dynamic_geometric_plan_time[length] = []
            dynamic_geometric_plan_time[length].append(segment['dynamic_geometric_plan_time'][t] + 1)

total_nlp_mean, total_nlp_std = [], []
index = [2, 4, 6, 8, 10, 12, 14, 16] 
for l in sorted(dynamic_solve_nlp):
    if l in index:
        total_nlp_mean.append(np.mean(dynamic_solve_nlp[l]))
        total_nlp_std.append(np.std(dynamic_solve_nlp[l]))
geo_time_mean, geo_time_std = [], []
for l in sorted(dynamic_geometric_plan_time):
    if l in index:
        geo_time_mean.append(np.mean(dynamic_geometric_plan_time[l]))
        geo_time_std.append(np.std(dynamic_geometric_plan_time[l]))
print('Single plan: ')
print(f'Success: {single_success / total}')
print(f'Symbolic plan time: {np.mean(single_symbolic_plan_time)} +- {np.std(single_symbolic_plan_time)}')
print(f'Task reduction: {np.mean(single_reduction_ratio)} +- {np.std(single_reduction_ratio)}')

print('Dynamic plan: ')
print(f'Success: {dynamic_success / total}')
print(f'Symbolic plan time: {np.mean(dynamic_symbolic_plan_time)} +- {np.std(dynamic_symbolic_plan_time)}')
print(f'Task reduction: {np.mean(dynamic_reduction_ratio)} +- {np.std(dynamic_reduction_ratio)}')
print(f'Path reduction: {np.mean(dynamic_path_reduction)} +- {np.std(dynamic_path_reduction)}')

plt.errorbar(index, total_nlp_mean, total_nlp_std, fmt='ok', lw=3)
plt.show()
plt.errorbar(index, geo_time_mean, geo_time_std, fmt='ok', lw=3)
plt.show()