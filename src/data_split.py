import pickle
import os
import torch

from data_tools import *

PROCESS_DIR = '/home/tometaro/Documents/Thesis/processed_data/processed/'
DATASET_DIR = '/home/tometaro/Documents/Thesis/datasets/VTNet/'

dataset=VTNet_Dataset(path=PROCESS_DIR)

with open(f'{DATASET_DIR}data.pkl', 'wb') as file:
    pickle.dump(dataset, file)


for i,(t,v,tt) in enumerate(dataset.doubleStratifiedSplit(downsample=True)):
    train_set = VTNet_Dataset(scanpaths=dataset.scanpaths[t,:,:,:], rawdata=dataset.rawdata[t], subject=dataset.subject[t], groups=dataset.groups[t])
    val_set = VTNet_Dataset(scanpaths=dataset.scanpaths[v,:,:,:], rawdata=dataset.rawdata[v], subject=dataset.subject[v], groups=dataset.groups[v])
    test_set = VTNet_Dataset(scanpaths=dataset.scanpaths[tt,:,:,:], rawdata=dataset.rawdata[tt], subject=dataset.subject[tt], groups=dataset.groups[tt])

    mean_pupil_1 = torch.mean(train_set.rawdata[:,:,3])
    std_pupil_1 = torch.std(train_set.rawdata[:,:,3])

    mean_pupil_2 = torch.mean(train_set.rawdata[:,:,6])
    std_pupil_2 = torch.std(train_set.rawdata[:,:,6])

    train_set.rawdata[:,:,3] = (train_set.rawdata[:,:,3]-mean_pupil_1)/std_pupil_1
    train_set.rawdata[:,:,6] = (train_set.rawdata[:,:,6]-mean_pupil_2)/std_pupil_2

    test_set.rawdata[:,:,3] = (test_set.rawdata[:,:,3]-mean_pupil_1)/std_pupil_1
    test_set.rawdata[:,:,6] = (test_set.rawdata[:,:,6]-mean_pupil_2)/std_pupil_2

    val_set.rawdata[:,:,3] = (val_set.rawdata[:,:,3]-mean_pupil_1)/std_pupil_1
    val_set.rawdata[:,:,6] = (val_set.rawdata[:,:,6]-mean_pupil_2)/std_pupil_2

    with open(f'{DATASET_DIR}trainset_vtnet_{i+1}.pkl', 'wb') as file:
        pickle.dump(train_set, file)
        
    with open(f'{DATASET_DIR}valset_vtnet_{i+1}.pkl', 'wb') as file:
        pickle.dump(val_set, file)
        
    with open(f'{DATASET_DIR}testset_vtnet_{i+1}.pkl', 'wb') as file:
        pickle.dump(test_set, file)