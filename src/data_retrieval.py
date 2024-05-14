# Fix randomness and hide warnings
seed = 42

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'

import numpy as np
np.random.seed(seed)

import pickle

from data_tools import *


DATA_DIR = '/home/tometaro/Documents/Thesis/data/'
UNPROCESS_DIR = '/home/tometaro/Documents/Thesis/processed_data/raw/'


for group in ['control', 'patient']:
    for subject in os.listdir(f'{DATA_DIR}tsv/{group}'):
        with open(f'{DATA_DIR}tsv/{group}/{subject}', 'r') as tsv:
            with open(f"{DATA_DIR}/behavioral/{group}/{subject[:-3]}csv", 'r') as behavioural_file:

                header = next(behavioural_file).replace('"','').split(',')

                age_ind = header.index('age')
                education_ind = header.index('education')
                group_ind = header.index('pdata')
                subject_ind = header.index('subject_nr')
                trial_ind = header.index('trial_seq')
                task_ind = header.index('task')

                correct_ind = header.index('correct_response')
                answear_ind = header.index('response_recall_image')

                imageR_ind = header.index('image_right')
                imageL_ind = header.index('image_left')

                imageLType_ind = header.index('memory_left')
                
                nlist_ind = header.index('nlist')

                data = {}
                starting = False
                time = -1
                for line in tsv:

                    l =line.split('\t')

                    if 'start_trial' in l[-1]:
                        rawData = []
                        behavioural_line = next(behavioural_file).replace('"','').split(',')

                        curr_Trial = int(behavioural_line[trial_ind])

                        if behavioural_line[task_ind] == "E":
                            second_time = 0 if subject.count('_')==1 else 0.5
                            data[curr_Trial] = Trial(behavioural_line[group_ind], int(behavioural_line[subject_ind])+second_time, 
                                                    int(behavioural_line[age_ind]), int(behavioural_line[education_ind]), behavioural_line[nlist_ind]+'_'+behavioural_line[trial_ind])

                        starting = True
                        time = None
                        continue

                    if 'end of' in l[-1]:
                        starting = False

                        if behavioural_line[task_ind] == "E":
                            data[curr_Trial].set_encoding(np.array(rawData, dtype='float64'))

                        elif behavioural_line[task_ind] == "R":
                            if behavioural_line[imageLType_ind]=="old":
                                is_right = False
                                im_old = behavioural_line[imageL_ind]
                                im_new = behavioural_line[imageR_ind]
                            else:
                                is_right = True
                                im_new = behavioural_line[imageL_ind]
                                im_old = behavioural_line[imageR_ind]

                            data[curr_Trial].set_recognition(np.array(rawData, dtype='float64'), im_old, im_new, behavioural_line[correct_ind]==behavioural_line[answear_ind], is_right)

                    if starting:
                        if time is None:
                            time = int(l[1])

                        if l[3] == '7':
                            avg = l[6:8]
                            
                            if behavioural_line[task_ind] == "E":
                                avg[0] = avg[0] if abs(float(avg[0])-800)<=350 else np.nan
                                avg[1] = avg[1] if abs(float(avg[1])-450)<=350 else np.nan
                            elif behavioural_line[task_ind] == "R":
                                avg[0] = avg[0] if abs(float(avg[0])-800)<=700 else np.nan
                                avg[1] = avg[1] if abs(float(avg[1])-450)<=280 else np.nan
                            
                            rawData+=[[int(l[1])-time]+avg+l[11:14]+l[18:21]]
                        else:
                            rawData+=[[int(l[1])-time]+[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]]

                with open(f'{UNPROCESS_DIR}{group}_{subject[:-3]}pkl', 'wb') as file:
                    pickle.dump(list(data.values()), file)