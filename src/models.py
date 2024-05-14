import torch
from torch import nn
from copy import deepcopy

class VETNet(nn.Module):
    def __init__(self, timeseries_size ,scanpath_size, cnn_shape = (16,6)):
        super(VETNet, self).__init__()
        self.scanpath_layer = nn.Sequential(
            nn.Conv2d(scanpath_size[0],cnn_shape[0],kernel_size=5,padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cnn_shape[0],cnn_shape[1],kernel_size=5,padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(scanpath_size[1]*scanpath_size[2]*cnn_shape[1]//16,50)
        )
        
        self.timeseries_layer_attention = nn.MultiheadAttention(timeseries_size[-1]-1,1, batch_first=True)
        self.timeseries_layer_gru = nn.GRU(input_size=timeseries_size[-1]-1, hidden_size=256, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(306,20),
            #nn.LeakyReLU(),
            nn.Linear(20,2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x_timeseries, x_scanpath):
        x_scanpath = self.scanpath_layer(x_scanpath)
        
        x_timeseries,_ = self.timeseries_layer_attention(x_timeseries,x_timeseries,x_timeseries)
        _ ,x_timeseries = self.timeseries_layer_gru(x_timeseries)
        
        x = torch.cat((torch.squeeze(x_timeseries,0),x_scanpath),dim=1)
        
        return self.classifier(x)

class EarlyStopping():
    def __init__(self, patience: int, mode: str, minimum_delta = 0.0):
        assert mode in {'min', 'max'}, "mode has to be 'min' or 'max'"
        self.minimum_delta = minimum_delta
        self.patience = patience
        self.counter = 0
        self.tracking = torch.inf if mode == 'min' else -torch.inf
        self.mode = 1 if mode=='max' else -1
        self.best_model = None
        
    def check(self, value, model: nn.Module):
        if self.mode*(value-self.tracking) <= self.minimum_delta:
            self.counter+=1
        else:
            self.counter = 0
            self.tracking = value
            self.best_model = deepcopy(model.state_dict())
            
        if self.counter == self.patience:
            model.load_state_dict(self.best_model)
            return True
        
        return False


class ReduceLROnPlateau():
    def __init__(self, patience: int, rate: float, mode: str, minimum_lr = 0.0, minimum_delta = 0.0):
        assert rate<=1 and rate>0, "rate as to be a number between 0 and 1"
        assert mode in {'min', 'max'}, "mode has to be 'min' or 'max'"

        self.minimum_delta = minimum_delta
        self.patience = patience
        self.counter = 0
        self.rate = rate
        self.minimum_lr = minimum_lr
        self.tracking = torch.inf if mode == 'min' else -torch.inf
        self.mode = 1 if mode=='max' else -1
        self.best_model = None
        
    def check(self, value, optimizer, model):
        if self.mode*(value-self.tracking) <= self.minimum_delta:
            self.counter+=1
        else:
            self.counter = 0
            self.tracking = value
            self.best_model = deepcopy(model.state_dict())
            
        if self.counter == self.patience:
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = max(self.rate*optimizer.param_groups[i]['lr'], self.minimum_lr)
            model.load_state_dict(self.best_model)
            self.counter = 0