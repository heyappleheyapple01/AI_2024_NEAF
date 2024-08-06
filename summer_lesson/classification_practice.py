import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt

## first part : Create data
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)   # torch.Size([100, 2])
y0 = torch.zeros(100)              # label = 0
x1 = torch.normal(-2 * n_data, 1)  # torch.Size([100, 2])
y1 = torch.ones(100)               # label = 1 
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # torch.Size([100, 2])
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # torch.Size([200])


## second part : Create NN

class Net(torch.nn.Module): # class a Network and input a torch module
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__() # To inherit things from Net, the standard process must be added
        self.hidden = torch.nn.Linear(n_feature, 10)  # hidden layer
        self.out = torch.nn.Linear(10, n_output)      # output layer
    def forward(self, x):
        x = F.relu(self.hidden(x))   # activation function for hidden layer
        x = self.out(x)              # output layer
        return x 

net = Net(n_feature=2, n_output=2) # define the network
print(net) # network architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss() # 已經包含了 sigmoid or softmax


plt.ion()

for t in range(100):
    
    out = net(x)                             # input x and predict based on x
    loss = loss_func(out, y)                 # compute the loss between prediction and true values

    optimizer.zero_grad()                    # clear gradients for next train
    loss.backward()                          # backpropagation, compute gradients
    optimizer.step()                         # apply gradients
    
    if t % 5 == 0:
        plt.cla()
        prediction = torch.max(out, 1)[1]    
        pred_y = prediction.data.numpy()     # convert tensor to numpy
        target_y = y.data.numpy()            # convert tensor to numpy
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)   # 預測正確除以總數量
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
        
plt.ioff()
plt.show()