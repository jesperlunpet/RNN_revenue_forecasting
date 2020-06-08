import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import csv
import torch.optim as optim
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

features = 16

x = [[ 1.680910e+05,0.000000e+00,1.251800e+05,2.064000e+04,3.419870e+05, 4.288100e+04,3.848680e+05,3.991820e+05,1.250000e+05, -1.138000e+03, 1.238620e+05,1.258160e+05,2.753200e+05,2.753200e+05,3.991820e+05, 0.000000e+00], [ 1.930280e+05,0.000000e+00,1.144320e+05,5.177230e+05,1.261570e+05, 6.730550e+05,6.755550e+05,1.250000e+05,0.000000e+00,1.390540e+05, 2.640540e+05,1.788260e+05,4.115010e+05,4.115010e+05,6.755550e+05, 0.000000e+00], [ 3.017540e+05,0.000000e+00,1.625940e+05,0.000000e+00,7.788220e+05, 4.957000e+05,1.303989e+06,1.413910e+06,1.250000e+05,0.000000e+00, 3.016490e+05,4.266490e+05,3.536870e+05,9.859640e+05,9.859640e+05, 1.413910e+06]]

def conv(x):
    conv1 = lambda i: i != 0 or False

    identifier = []
    for i in x:
        identifier.append([conv1(j) for j in i])

    x = np.asarray(x)

    mean = np.mean(x, axis = 0)
    std = np.std(x, axis = 0)

    x -= np.mean(x, axis = 0)
    x /= np.std(x, axis = 0)
    x[np.isnan(x)] = 0.0

    x = np.concatenate((x, np.asarray(identifier)), axis=1)

    return x, mean, std

in_features = features*2
hidden_dim = 50
out_features = features*2
n_layers = 2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.lstm = nn.LSTM(in_features, hidden_dim, num_layers = n_layers, batch_first = True)
        # self.gru = nn.GRU(in_features, hidden_dim, num_layers = n_layers, batch_first = True)
        self.l_out = nn.Linear(in_features = hidden_dim, out_features = out_features, bias = False)        

    def forward(self, x):
        h0 = torch.zeros(n_layers, 1, hidden_dim).to(device)
        c0 = torch.zeros(n_layers, 1, hidden_dim).to(device)
        x, _ = self.lstm(x, (h0,c0))
        # x, _ = self.gru(x, h0)
        x = x.view(-1, hidden_dim)
        x = self.l_out(x)
        return x[-1]

net = Net().to(device)

net.load_state_dict(torch.load('model.pt', map_location='cpu'))

net.eval()

X, mean, std = conv(x)

target = X[-1, :]
inputs = X[:-1, :]

inputs = torch.Tensor(inputs)
inputs = inputs.reshape(1, inputs.size(0), in_features)

target = torch.Tensor(target)

outputs = net.forward(inputs)

print((outputs.detach().cpu().numpy()[:16] * std) + mean)