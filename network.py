import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import csv
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

# DEVICE CONFIGURATION

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DATA PREPARATION

csv = csv.reader(open('shuffled_data.csv'), delimiter=',')

features = 16

data = []
for i in csv:
    # Remove date as we don't want date to influence training
    for j in range(int(len(i)/22)):
        del i[(j*17)-j:((j*17)-j+5)]
        del i[(j*10)-j]
        # del i[(j*22)-j]
    # Append identifier list for missing data

    conv1 = lambda i: i != "" or False

    identifier = ([conv1(j) for j in i])

    # Convert list to list of integers
    conv2 = lambda i: i or 0
    i = [conv2(j) for j in i]
    i = list(map(float, i))

    # Convert 1d to 2d array with 22 columns
    i = [i[j:j+features] for j in range(0, len(i), features)]
    identifier = [identifier[j:j+features] for j in range(0, len(identifier), features)]

    dataset = np.asarray(i)

    # Remove too small data samples
    if (dataset.ndim == 1 or len(dataset) < 3 or len(dataset) > 8):
        continue
    dataset = dataset[:,0:features]

    # Normalization
    dataset -= np.mean(dataset, axis = 0)
    dataset /= np.std(dataset, axis = 0)
    # As division of 0 results in NaN, it is replaced by 0.
    # This occurs when the sequence of one feature is always 0.
    dataset[np.isnan(dataset)] = 0.0

    dataset = np.concatenate((dataset, np.asarray(identifier)), axis=1)

    data.append(dataset)

training = data[:int(len(data)*0.8)]
validation = data[-int(len(data)*0.2):-int(len(data)*0.1)]
testing = data[-int(len(data)*0.1):]

print(len(training))
print(len(validation))
print(len(testing))

# NEURAL NETWORK

in_features = features*2
hidden_dim = 50
out_features = features*2
n_layers = 2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.h0 = torch.zeros(n_layers, 1, hidden_dim).to(device)
        self.h0 = torch.zeros(n_layers, 1, hidden_dim).to(device)

        self.lstm = nn.LSTM(in_features, hidden_dim, num_layers = n_layers, batch_first = True, dropout=0.5)
        # self.gru = nn.GRU(in_features, hidden_dim, num_layers = n_layers, batch_first = True, dropout=0.5)
        self.l_out = nn.Linear(in_features = hidden_dim, out_features = out_features, bias = False)

    def reset_hidden_state(self):
        self.h0 = torch.zeros(n_layers, 1, hidden_dim).to(device)
        self.h0 = torch.zeros(n_layers, 1, hidden_dim).to(device)

    def forward(self, x):
        self.reset_hidden_state()
        x, _ = self.lstm(x, (self.h0, self.c0))
        x = x.view(self.lstm.hidden_size)
        x = self.l_out(x)
        return x[-1]

net = Net().to(device)
print(net)

# TRAINING AND VALIDATION

num_epochs = 20
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.05, momentum = 0.8, weight_decay = 1e-5)
scheduler = StepLR(optimizer, step_size=2, gamma=0.96)

training_loss, validation_loss, accuracy1, accuracy2 = [], [], [], []
# Theshold for accuracy at around the market rate
threshold1 = 0.05
threshold2 = 0.10

for i in range(num_epochs):

    epoch_training_loss = 0
    epoch_validation_loss = 0
    total_num_corrects1 = 0
    total_num_corrects2 = 0

    np.random.shuffle(training)
    
    net.eval()
    
    for batch in validation:
        target = batch[-1,:]
        inputs = batch[:-1,:]

        inputs = torch.Tensor(inputs).to(device)
        inputs = inputs.reshape(inputs.size(0), 1, in_features)
        target = torch.Tensor(target).to(device)

        outputs = net(inputs)
        loss = criterion(outputs, target)
        
        epoch_validation_loss += loss.detach().cpu().numpy()

        pred = (((outputs - target) >= -threshold1) & ((outputs - target) <= threshold1)).view_as(target)  # to make pred have same shape as target
        num_correct = torch.sum(pred).item()
        total_num_corrects1 += num_correct
        pred = (((outputs - target) >= -threshold2) & ((outputs - target) <= threshold2)).view_as(target)  # to make pred have same shape as target
        num_correct = torch.sum(pred).item()
        total_num_corrects2 += num_correct
    
    net.train()
    
    for batch in training:
        target = batch[-1,:]
        inputs = batch[:-1,:]

        inputs = torch.Tensor(inputs).to(device)
        inputs = inputs.reshape(inputs.size(0), 1, in_features)
        target = torch.Tensor(target).to(device)
        
        outputs = net(inputs)
        loss = criterion(outputs, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_training_loss += loss.detach().cpu().numpy()

    scheduler.step()

    # Save loss for plot
    training_loss.append(epoch_training_loss/len(training))
    validation_loss.append(epoch_validation_loss/len(validation))

    # Save accuracy for plot
    accuracy1.append(total_num_corrects1/len(testing))
    accuracy2.append(total_num_corrects2/len(testing))

    # Print loss every 5 epochs
    if i % 1 == 0:
        print('Epoch '+str(i)+', learning rate: '+str(scheduler.get_lr())+' training loss: '+str(training_loss[-1])+', validation loss: '+str(validation_loss[-1])+', 0.05 accuracy: '+str(accuracy1[-1])+', 0.10 accuracy: '+str(accuracy2[-1]))

torch.save(net.state_dict(), 'test.pt')

net.eval()

total_num_corrects1 = 0
total_num_corrects2 = 0
correct1, correct2 = [], []

for batch in testing:
    target = batch[-1, :]
    inputs = batch[:-1, :]

    # Convert inputs to tensor
    inputs = torch.Tensor(inputs).to(device)
    inputs = inputs.reshape(inputs.size(0), 1, in_features)

    # Convert target to tensor
    target = torch.Tensor(target).to(device)

    # Forward pass
    outputs = net.forward(inputs)
    
    # Prediction is compared to a threshold as 100 % accuracy is not needed
    pred = (((outputs - target) >= -threshold1) & ((outputs - target) <= threshold1)).view_as(target)  # to make pred have same shape as target
    num_correct = torch.sum(pred).item()
    total_num_corrects1 += num_correct
    correct1.append(num_correct)
    pred = (((outputs - target) >= -threshold2) & ((outputs - target) <= threshold2)).view_as(target)  # to make pred have same shape as target
    num_correct = torch.sum(pred).item()
    total_num_corrects2 += num_correct
    correct2.append(num_correct)

print('Test 0.05 accuracy is '+str(total_num_corrects1/len(testing)))
print('Test 0.10 accuracy is '+str(total_num_corrects2/len(testing)))

test = testing[0]

target = batch[1:, :]
inputs = batch[:-1, :]

inputs = torch.Tensor(inputs)
inputs = inputs.reshape(inputs.size(0), 1, in_features)

target = torch.Tensor(target)

outputs = net.forward(inputs)

print(inputs)
print(target)
print(outputs)

plt.figure()
plt.plot(correct1, '.', label='Prediction')
plt.plot(correct2, '.', label='Prediction')
plt.legend()
plt.show()

epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, accuracy1, 'g', label='Accuracy',)
plt.plot(epoch, accuracy2, 'g', label='Accuracy',)
plt.xlabel('Epoch'), plt.ylabel('Accuracy')
plt.show()

plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss',)
plt.plot(epoch, validation_loss, 'b', label='Validation loss',)
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()