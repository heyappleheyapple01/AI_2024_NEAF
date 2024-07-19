import torch 
import torch.nn.functional as F 
import matplotlib.pyplot as plt

## first part : Create data
"""
your code
"""

## second part : Create NN
"""
basic form:
class Net(torch.nn.Module): # class a Network and input a torch module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() # To inherit things from Net, the standard process must be added

    def forward(self, x):
        
        return x 

optimizer = torch.optim.SGD()
loss_func = torch.nn.MSELoss()
"""

## third part : training
"""
your code
"""