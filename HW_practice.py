import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
from os import walk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix

import torch
import torch.optim as optim


def compute_windowed_std(signal, window_size, overlap):
    step = window_size - overlap
    windows = [signal[i:i+window_size] for i in range(0, len(signal)-window_size+1, step)]
    std_features = [np.std(window) for window in windows]
    return std_features

train_x = []
train_y = []
folder_name = ['Yes', 'No']

output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

i = 0
window_size = 256
overlap = 128

for folder in folder_name:
    path = os.path.join(r'C:\Users\goldenbullhornPC\OneDrive\Documents\university\Master\Learning\AI\Data', folder)  # Join the directory names correctly
    for root, dirs, files in os.walk(path):
        for f in files:
            filename = os.path.join(root, f)  # Construct full path to the file
            #print(filename)           
            
            acc = scipy.io.loadmat(filename)
            acc = acc['tsDS'][:,1].tolist()[0:7500]
            
            # Compute the windowed standard deviation of the signal
            std_features = compute_windowed_std(acc, window_size, overlap)
            
            train_x.append(std_features)

            if folder == 'Yes':    
                train_y.append(1)
                title = 'Original Signal With Chatter #'
                saved_file_name = 'Data'
            
            if folder == 'No':
                train_y.append(0)
                title = 'Original Signal Without Chatter #'
                saved_file_name = 'Data'
                
            #plt.clf()
            # plt.figure(figsize=(7,4))
            # plt.plot(acc, 'b-', lw=1)
            # plt.title(title + str(i+1))
            # plt.xlabel('Samples')
            # plt.ylabel('Acceleration')
            # output_path = os.path.join(output_dir, title.replace(' ', '_') + str(i + 1) + '.png')
            # plt.savefig(output_path)   
            #plt.show()
            i = i + 1

train_x = np.array(train_x)
train_y = np.array(train_y)

# Define the neural network
class Net(torch.nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 20)  # Input layer to hidden layer
        self.fc2 = torch.nn.Linear(20, 2)  # Hidden layer to output layer with 2 neurons
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # No activation function here, CrossEntropyLoss will handle it
        return x

scaler = MinMaxScaler(feature_range=(0, 1))
train_x = scaler.fit_transform(train_x)

train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.long)  # Use long type for classification

loo = LeaveOneOut()
y_pred = []
y_true = []

input_size = train_x.shape[1]  # Number of features from windowed standard deviation

print(train_x.shape)
print(input_size)

for train_idx, test_idx in loo.split(train_x):
    X_train, X_test = train_x[train_idx], train_x[test_idx]
    y_train, y_test = train_y[train_idx], train_y[test_idx]
    
    model = Net(input_size=input_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        predicted = torch.argmax(output, dim=1)
        y_pred.append(predicted.item())
        y_true.append(y_test.item())

# Calculate confusion matrix and accuracy
cf_m = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cf_m.ravel()
accuracy = (tn + tp) / (tn + fp + fn + tp)

print('Predictions: \t', y_pred)
print('Ground Truth: \t', y_true)
print('Confusion Matrix: \n', cf_m)
print('Accuracy: \t', accuracy)
