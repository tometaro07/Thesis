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

import random
random.seed(seed)

# Import tensorflow
import torch
import torch.version
from torch.utils.data import DataLoader
import torch.optim as optim


torch.manual_seed(seed)

torch.version.__version__
print('PyTorch Version:',torch.version.__version__)
print('Cuda Version:',torch.version.cuda,'\n')

print('Available devices:')
for i in range(torch.cuda.device_count()):
   print('\t',torch.cuda.get_device_properties(i).name)
   print('\t\tMultiprocessor Count:',torch.cuda.get_device_properties(i).multi_processor_count)
   print('\t\tTotal Memory:',torch.cuda.get_device_properties(i).total_memory/1024/1024, 'MB')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\n',device)

# Import other libraries
import matplotlib.pyplot as plt
import cv2
from skimage import transform
import pickle
from data_tools import *
from models import *
from tqdm import tqdm


DATASETS_DIR = '/home/tometaro/Documents/Thesis/datasets/VTNet/'

batchsize = 32
classes = ['CONTROL', 'PATIENT']

for j in range(1,20):
    for i in range(1,20):
        TP = 0
        FN = 0
        FP = 0
        TN = 0

        for k in range(1,6):
            
            # Load Data 
            with open(f'{DATASETS_DIR}trainset_vtnet_{k}.pkl', 'rb') as file:
                train_set = pickle.load(file)

            with open(f'{DATASETS_DIR}valset_vtnet_{k}.pkl', 'rb') as file:
                val_set = pickle.load(file)

            with open(f'{DATASETS_DIR}testset_vtnet_{k}.pkl', 'rb') as file:
                test_set = pickle.load(file)
                
            trainloader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
            valloader = DataLoader(val_set, batch_size=batchsize, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=batchsize, shuffle=False)
            
            # Create Model
            model = VETNet(timeseries_size=train_set[0][0].shape, scanpath_size=train_set[0][1].shape,cnn_shape=(i,j)).to(device)
            
            # Set CallBacks and Optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-5)

            val_criterion = nn.CrossEntropyLoss()
            lr_tracker = ReduceLROnPlateau(5, 0.5, mode='min', minimum_lr=1e-6)
            earlystop_tracker = EarlyStopping(10, mode='min')
            
            # Train Model
            
            running_loss = []
            val_running_loss = []
            for epoch in range(1,101):  # loop over the dataset multiple times

                running_loss += [0.0]
                val_running_loss += [0.0]
                
                for input_rawdata, input_scanpath, labels in trainloader:
                    # get the inputs; data is a list of [inputs, labels]
                    # tepoch.set_description(f"Epoch {epoch}")
                    input_rawdata = input_rawdata[:,:,1:].to(device)
                    input_scanpath = (input_scanpath/128-1).to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = model(input_rawdata, input_scanpath)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss[-1] += loss.item()
                    
                    # tepoch.set_postfix(loss=running_loss[-1])
            
                for val_rawdata, val_scanpath, val_labels in valloader:
                    val_rawdata = val_rawdata.to(device)
                    val_scanpath = (val_scanpath/128-1).to(device)
                    val_labels = val_labels.to(device)
                    
                    val_outputs = model(val_rawdata[:,:,1:], val_scanpath)
                    val_loss = val_criterion(val_outputs, val_labels)
                    val_running_loss[-1] += val_loss.item()
                
                # print(f"\t Training Loss (final): {running_loss[-1]/len(train_set): .4f}, Validation Loss: {val_running_loss[-1]/len(val_set): .4f}, Learning Rate: {optimizer.param_groups[-1]['lr']: .2E}")
                
                lr_tracker.check(value=val_running_loss[-1], optimizer=optimizer, model=model)
                
                if earlystop_tracker.check(value=val_running_loss[-1], model=model):
                    break
            
            # Test Results
            
            with torch.no_grad():
                for input_rawdata, input_scanpath, labels in test_loader:
                    
                    input_rawdata = input_rawdata[:,:,1:].to(device)
                    input_scanpath = (input_scanpath/128-1).to(device)
                    labels = labels.to(device)
                    outputs = model(input_rawdata, input_scanpath)
                    # max returns (value ,index)
                    _, predicted = torch.max(outputs, 1)
                    TP += torch.sum((predicted==0)[labels==0])
                    FN += torch.sum((predicted==1)[labels==0])
                    FP += torch.sum((predicted==0)[labels==1])
                    TN += torch.sum((predicted==1)[labels==1])


        # Test Results

        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        print(i,j)
        print(f'Sensitivity: {sensitivity*100} %')
        print(f'Specificity: {specificity*100} %')