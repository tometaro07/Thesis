# Fix randomness and hide warnings
seed = 42

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
np.random.seed(seed)

import logging

import pickle
import os

from data_tools import *

PROCCESS_DIR = '/home/tometaro/Documents/Thesis/processed_data/processed/'
UNPROCESS_DIR = '/home/tometaro/Documents/Thesis/processed_data/raw/'

# LOAD DATA

data = []

for file in os.listdir(UNPROCESS_DIR):
    with open(UNPROCESS_DIR+file, 'rb+') as f:
        data += [pickle.load(f)]


# REMOVE MISSING VALUES

encoding=''
recognition = ''
for i in range(len(data)):
    for j in range(len(data[i])):
        
        guide = np.any(np.isnan(data[i][j].encoding),axis=1)
        
        if np.sum(guide)>len(data[i][j].encoding)*0.2:
            data[i][j].encoding = None
            encoding+= f'{data[i][j].group} {data[i][j].subject} {data[i][j].trial}\n'
        else:
            data[i][j].encoding=data[i][j].encoding[~guide]

        guide = np.any(np.isnan(data[i][j].recognition),axis=1)
        
        if np.sum(guide)>len(data[i][j].recognition)*0.2:
            data[i][j].recognition = None
            recognition+= f'{data[i][j].group} {data[i][j].subject} {data[i][j].trial}\n'
        else:
            data[i][j].recognition=data[i][j].recognition[~guide]

with open("enconding_removed.txt", 'w') as output:
    output.write(encoding)
    
with open("recognition_removed.txt", 'w') as output:
    output.write(recognition)


# REMOVE BEST AND WORST PERFORMERS

results={'control':{},'patient': {}}

for i in range(len(data)):
    group = data[i][0].group
    subject = data[i][0].subject
    for j in range(len(data[i])):
        trial = data[i][j].trial.split('_')[1]
        if trial not in list(results[group].keys()):
            results[group][trial] = []
        results[group][trial] += [data[i][j].isCorrect]

for g in results.keys():
    for t in results[g].keys():
        results[g][t]=np.mean(results[g][t])

worst_trials_control = [k for k,v in results['control'].items() if v ==np.min(list(results['control'].values()))]
best_trials_patient = [k for k,v in results['patient'].items() if v ==np.max(list(results['patient'].values()))]

print(worst_trials_control, np.min(list(results['control'].values())))
print(best_trials_patient, np.max(list(results['patient'].values())))


for i in range(len(data)):
    indexes = []
    for j in range(len(data[i])):
        if data[i][j].trial.split('_')[1] in worst_trials_control+best_trials_patient:
            indexes = [j] + indexes
    
    for j in indexes:
        data[i].pop(j)
        

# SAVE DATA

for d in data:
    group=d[0].group
    subject=int(d[0].subject//1)
    alter = '' if d[0].subject%1 == 0 else '_5'
    # directory = 'first/' if d[0].subject%1==0 else 'second/'
    directory = ''
    
    for i in range(len(d)):
        if d[i].encoding is not None:
            d[i].build_scanpath(resize=(512,512))
            d[i].build_heatmap(distance=57, angle=1, resize=(512,512))
    with open(f'{PROCCESS_DIR}{directory}{group}_s{subject}{alter}.pkl', 'wb') as file:
        pickle.dump(d, file)