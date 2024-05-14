from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor
import pickle
import os
import numpy as np
import cv2
from sklearn.model_selection import StratifiedGroupKFold

class Trial:
    
    def __init__(self, group: str, subject: float, age: int, education: int, trial: str) -> None:
        
        self.group = group
        self.subject = subject
        self.age = age
        self.education = education
        self.trial = trial
        
        self.encoding = None
        self.recognition = None
        
        self.encodingScanPath = None
        self.recognitionScanPath = None
        
        self.encodingHeatMap = None
        self.recognitionHeatMap = None
        
        self.oldImage = None
        self.newImage = None
        self.isCorrect = None
        self.isOldRight = None
        
    def set_encoding(self, data):
        self.encoding = data
    
    def set_recognition(self, data, oldImage : str, newImage : str, isCorrect : bool, isOldRight : bool):
        
        self.recognition = data
        
        self.oldImage = oldImage
        self.newImage = newImage
        self.isCorrect = isCorrect
        self.isOldRight = isOldRight
        
    
    def build_scanpath(self, line_width = 5, point_size = 7, isEncoding = True, resize = None):
        
        if isEncoding:
            coords = list(zip(self.encoding[:,1]-450,self.encoding[:,2]-100))
        else:
            coords = list(zip(self.recognition[:,1]-100,self.recognition[:,2]-170))
    
        size = (700,700) if isEncoding else (1400,560)
    
        scan_path = Image.new(mode='L',size=size,color=255)
        draw = ImageDraw.Draw(scan_path)
        draw.line(xy=coords, 
                fill=0, width = line_width)
        
        for coord in coords:
            draw.regular_polygon(bounding_circle=(coord,point_size),
                    n_sides=20 , fill=0, width = point_size)
        
        scan_path = scan_path.resize(resize) if resize else scan_path
        
        if isEncoding:
            self.encodingScanPath = scan_path
        else:
            self.recognitionScanPath = scan_path
        
    def build_heatmap(self, distance, angle, isEncoding = True,resize = None):
        def multivariate_gaussian(shape, mu, Sigma):
            """Return the multivariate Gaussian distribution on array pos."""

            X = np.linspace(-shape[0]/2, shape[0]/2, shape[0])
            Y = np.linspace(-shape[1]/2, shape[0]/2, shape[1])
            X, Y = np.meshgrid(X, Y)
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X
            pos[:, :, 1] = Y
            
            n = len(mu)
            Sigma_det = np.linalg.det(Sigma)
            Sigma_inv = np.linalg.inv(Sigma)
            N = np.sqrt((2*np.pi)**n * Sigma_det)
            N=1
            # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
            # way across all the input variables.
            fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

            return np.exp(-fac / 2) / N
            
        if isEncoding:
            coords = list(zip(self.encoding[:,1]-800,self.encoding[:,2]-450))
        else:
            coords = list(zip(self.recognition[:,1]-800,self.recognition[:,2]-450))
    
        size = (700,700) if isEncoding else (1400,560)
        Sigma = np.array([[ 1800*np.tan(angle*np.pi/180)*distance/33.8 , 0], [0, 900*np.tan(angle*np.pi/180)*distance/27.1]])
        
        heatmap = np.zeros(shape=size)
        
        for coord in coords:
            heatmap += multivariate_gaussian(size,coord,Sigma)
            
        heatmap = cv2.resize(heatmap, dsize=resize) if resize else heatmap
        
        if isEncoding:
            self.encodingHeatMap = heatmap
        else:
            self.recognitionHeatMap = heatmap
        

class VTNet_Dataset (Dataset):

    def __init__(self, path='', scanpaths = [], rawdata = [], groups = [], subject = [], useHeatmap = False):
        
        if len(scanpaths)==0:

            self.scanpaths = []
            self.rawdata = []
            self.groups = []
            self.subject = []
            
            
            for file in [x for x in os.listdir(path) if x[-3:]=='pkl']:
                with open(path+file, 'rb+') as f:
                    if not useHeatmap:
                        self.scanpaths += list(map(lambda x: x.encodingScanPath or None, pickle.load(f)))
                    else:
                        self.scanpaths += list(map(lambda x: x.encodingHeatMap or None, pickle.load(f)))
                with open(path+file, 'rb+') as f:
                    self.rawdata += list(map(lambda x: x.encoding[:,[0,3,4,5,6,7,8]] if x.encodingScanPath else None,pickle.load(f)))
                with open(path+file, 'rb+') as f:
                    self.groups += list(map(lambda x: (0 if x.group[0]=='c' else 1) if x.encodingScanPath else None,pickle.load(f)))
                with open(path+file, 'rb+') as f:
                    self.subject += list(map(lambda x: x.subject if x.encodingScanPath else None,pickle.load(f)))

            self.groups = np.array(self.groups)

            def new_len(x):
                return len(x) if hasattr(x, '__iter__') else 0

            max_len = max(map(new_len,self.rawdata))
            self.rawdata = np.array([np.pad(rd,((max_len-len(rd),0),(0,0)),constant_values=0) if hasattr(rd, '__iter__') else None for rd in self.rawdata], dtype=object)[self.groups!=None]
            self.rawdata = torch.tensor(np.array(list(self.rawdata)), dtype=torch.float32)
            
            
            self.rawdata[:,:,0] = self.rawdata[:,:,0]/1500 - 1
            self.rawdata[:,:,[1,4]] = (self.rawdata[:,:,[1,4]]-800)/350
            self.rawdata[:,:,[2,5]] = (self.rawdata[:,:,[2,5]]-450)/350
            
            
            self.subject = torch.tensor(np.array(self.subject)[self.groups!=None].astype(int))
            self.scanpaths = np.array(self.scanpaths, dtype=object)[self.groups!=None]
            self.scanpaths=[pil_to_tensor(rd) for rd in self.scanpaths] if not useHeatmap else self.scanpaths
            self.scanpaths = torch.stack(self.scanpaths, dim=0)
            self.groups = torch.tensor(self.groups[self.groups!=None].astype(int))

        else:
            self.scanpaths = scanpaths
            self.rawdata = rawdata
            self.groups = groups
            self.subject = subject

        print(torch.sum(self.groups==0),torch.sum(self.groups==1))

    def __getitem__(self, index):
        return self.rawdata[index], self.scanpaths[index], self.groups[index]
    
    def __len__(self):
        return len(self.groups)
    
    def doubleStratifiedSplit(self, split_fractions = 0.8, downsample=False):
        
        ind = np.arange(len(self.groups))
        
        if downsample:
            aux = self.groups==0 if torch.sum(self.groups==0)>=self.__len__()/2 else self.groups==1
            aux = ind[aux]
            aux = ind[aux[torch.randperm(aux.shape[0])[len(self.groups)-len(aux):]]]
            mask = torch.ones(len(ind),dtype=torch.bool)
            mask[aux] = False
            ind = ind[mask]

        sub_group = (self.groups[ind]*2-1)*int(self.subject[ind])
        
        sgkf = StratifiedGroupKFold(n_splits=int(1/(1-split_fractions)), random_state=None, shuffle=False)
        
        output = []
        
        for trainset_ind, aux_ind in sgkf.split(ind, self.groups[ind], sub_group):
        
            sgkf = StratifiedGroupKFold(n_splits=2, random_state=None, shuffle=False)
            
            valset_ind, testset_ind = list(sgkf.split(ind[aux_ind], self.groups[aux_ind], sub_group[aux_ind]))[0]
            
            output += [(ind[trainset_ind], ind[aux_ind[valset_ind]], ind[aux_ind[testset_ind]])]
        
        return output