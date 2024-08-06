import numpy as np
import torch ## The function of torch is the same as numpy, but it can run on GPU

## first part : convert data type
np_data = np.arange(6).reshape((2, 3)) 
torch_data = torch.from_numpy(np_data)
print(f"Tensor :\n{torch_data}")
tensor2array = torch_data.numpy()

## second part : some basic usage