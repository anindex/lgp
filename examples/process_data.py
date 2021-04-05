import pickle
import sys
import argparse
import numpy as np
import pandas as pd
from os.path import join, dirname, abspath, expanduser
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT_DIR = join(dirname(abspath(__file__)), '..')
DATA_DIR = join(ROOT_DIR, 'data', 'experiments')
sys.path.append(ROOT_DIR)

from lgp.geometry.trajectory import compute_path_length

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python process_data.py data.p')
parser.add_argument('--name', help='The scenario name of the domain, problem file', type=str, default='prediction.p')
args = parser.parse_args()
data_file = join(DATA_DIR, args.name)

with open(data_file, 'rb') as f:
    data = pickle.load(f)

total = len(data)
print('Total data: ', total)
single_success, dynamic_success = 0, 0
single_reduction_ratio, dynamic_reduction_ratio = [], []
single_symbolic_plan_time, dynamic_symbolic_plan_time = [], []
dynamic_num_change_plan = []
dynamic_path_reduction = []
dynamic_solve_nlp = {}
dynamic_geometric_plan_time = {}
dynamic_geometric_plan_time_over_task = {}
dynamic_path_len_over_task = {}
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
        dynamic_num_change_plan.append(segment['dynamic_num_change_plan'])
    if segment['dynamic_success'] and segment['single_success']:
        dynamic_path_reduction.append(dynamic_path / single_path)
    for t in segment['dynamic_num_failed_plans']:
        length = len(segment['dynamic_plans'][t][0])
        if length not in dynamic_solve_nlp:
            dynamic_solve_nlp[length] = []
        dynamic_solve_nlp[length].append(segment['dynamic_num_failed_plans'][t] + 1)
    max_t = max(segment['dynamic_geometric_plan_time'].keys())
    for t in segment['dynamic_geometric_plan_time']:
        if t in segment['dynamic_plans']:
            length = len(segment['dynamic_plans'][t][0])
            if length not in dynamic_geometric_plan_time:
                dynamic_geometric_plan_time[length] = []
            dynamic_geometric_plan_time[length].append(segment['dynamic_geometric_plan_time'][t] + 1)
        i = int(round(t / max_t * 10))
        if i not in dynamic_geometric_plan_time_over_task:
            dynamic_geometric_plan_time_over_task[i] = []
        dynamic_geometric_plan_time_over_task[i].append(segment['dynamic_geometric_plan_time'][t])
    max_t = max(segment['dynamic_plans'].keys())
    for t in segment['dynamic_plans']:
        i = int(round(t / max_t * 10))
        if i not in dynamic_path_len_over_task:
            dynamic_path_len_over_task[i] = []
        dynamic_path_len_over_task[i].append(len(segment['dynamic_plans'][t][0]))
total_nlp_data = []
max_length = max(dynamic_geometric_plan_time.keys())
index = list(range(2, max_length + 1, 2))
for l in sorted(dynamic_solve_nlp):
    if l in index:
        total_nlp_data.append(dynamic_solve_nlp[l])
geo_time_data = []
for l in sorted(dynamic_geometric_plan_time):
    if l in index:
        geo_time_data.append(dynamic_geometric_plan_time[l])
dynamic_geometric_plan_time.pop(5, None)
dynamic_geometric_plan_time.pop(9, None)
dynamic_solve_nlp.pop(5, None)
dynamic_solve_nlp.pop(9, None)
print('Single plan: ')
print(f'Success: {single_success / total}')
print(f'Symbolic plan time: {np.mean(single_symbolic_plan_time)} +- {np.std(single_symbolic_plan_time)}')
print(f'Task reduction: {np.mean(single_reduction_ratio)} +- {np.std(single_reduction_ratio)}')

print('Dynamic plan: ')
print(f'Success: {dynamic_success / total}')
print(f'Symbolic plan time: {np.mean(dynamic_symbolic_plan_time)} +- {np.std(dynamic_symbolic_plan_time)}')
print(f'Num change plan: {np.mean(dynamic_num_change_plan)} +- {np.std(dynamic_num_change_plan)}')
print(f'Task reduction: {np.mean(dynamic_reduction_ratio)} +- {np.std(dynamic_reduction_ratio)}')
print(f'Path reduction: {np.mean(dynamic_path_reduction)} +- {np.std(dynamic_path_reduction)}')

portion = [i / 10 for i in dynamic_geometric_plan_time_over_task]
data = dynamic_geometric_plan_time_over_task.values()
fig, ax = plt.subplots()
ax.boxplot(data)
ax.set_xticklabels(portion)
ax.set_xlabel('Task progress ratio')
ax.set_ylabel('Total solution time (s)')
plt.show()

portion = sorted([i / 10 for i in dynamic_path_len_over_task])
portion.pop()
data = [dynamic_path_len_over_task[i] for i in sorted(dynamic_path_len_over_task.keys())]
data.pop()
fig, ax = plt.subplots()
ax.boxplot(data)
ax.set_xticklabels(portion)
ax.set_xlabel('Task progress ratio')
ax.set_ylabel('Skeleton length')
plt.show()

fig, ax = plt.subplots()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.boxplot(total_nlp_data)
ax.set_xticklabels(index)
ax.set_xlabel('Skeleton length')
ax.set_ylabel('Number of NLP solved')
plt.show()
fig, ax = plt.subplots()
ax.boxplot(geo_time_data)
ax.set_xticklabels(index)
ax.set_xlabel('Skeleton length')
ax.set_ylabel('Total solution time (s)')
plt.show()
