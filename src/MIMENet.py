import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm

# MIMENet Deep Neural Network
class MIMENet(nn.Module):
    # Constructor for 5 fully connected layers with bottleneck for layers 4 and 5
    def __init__(self, input_size, hidden_size_factor, bottleneck, output_size):
        super(MIMENet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_factor*input_size)
        self.fc2 = nn.Linear(hidden_size_factor*input_size, hidden_size_factor*input_size)
        self.fc3 = nn.Linear(hidden_size_factor*input_size, hidden_size_factor*input_size)
        self.fc4 = nn.Linear(hidden_size_factor*input_size, int(hidden_size_factor*input_size*bottleneck))
        self.fc5 = nn.Linear(int(hidden_size_factor*input_size*bottleneck), output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # Forward pass
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x

#custom train loader
class CustomTrainSet(torch.utils.data.Dataset):
    #read data from text file
    #split data into input all columns except last one and output last column
    def __init__(self, path):
        data = np.loadtxt(path)
        self.x = torch.from_numpy(data[:, :-1]).float()
        #ensure shape is (n, 1)
        self.y = torch.from_numpy(data[:, -1]).float().view(-1, 1)
        self.len = data.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

#custom test loader
class CustomTestSet(torch.utils.data.Dataset):
    #read data from text file
    #split data into input all columns except last one and output last column
    def __init__(self, path):
        data = np.loadtxt(path)
        self.x = torch.from_numpy(data[:, :-1]).float()
        #ensure shape is (n, 1)
        self.y = torch.from_numpy(data[:, -1]).float().view(-1, 1)
        self.len = data.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


#training function
def train(training_path, test_path, epochs, learning_rate, batch_size, lambda_l1, hidden_size_factor, bottleneck):

    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print device
    print("Device: " + str(device))

    #custom train loader
    train_loader = torch.utils.data.DataLoader(dataset=CustomTrainSet(training_path), batch_size=batch_size, shuffle=True)

    #custom test loader
    test_loader = torch.utils.data.DataLoader(dataset=CustomTestSet(test_path), batch_size=batch_size, shuffle=True)

    #get input size
    input_size = len(open(training_path).readline().split(' ')) - 1

    #get output size
    output_size = 1

    #initialize model
    model = MIMENet(input_size, hidden_size_factor, bottleneck, output_size)

    #move model to device
    model.to(device)

    #custom loss function
    def custom_loss(output, target, lambda_l1, model):
        #l1 regularization
        l1_reg = 0
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)
        
        #return binary cross entropy loss + l1 regularization
        return F.binary_cross_entropy(output, target, reduction='mean') + lambda_l1*l1_reg

    #mean absolute error
    def mae(output, target):
        return F.l1_loss(output, target, reduction='mean')

    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #training history
    train_history = []

    #mae evaluation history for training set
    mae_history = []

    #training loop
    for epoch in range(epochs):

        print("Epoch: " + str(epoch+1) + "/" + str(epochs))

        #training loop
        for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            #cast to device
            x = x.to(device)
            y = y.to(device)

            #forward pass
            output = model(x)
            #calculate loss
            loss = custom_loss(output, y, lambda_l1, model)
            #clear gradients
            optimizer.zero_grad()
            #backward pass
            loss.backward()
            #update weights
            optimizer.step()
            #append loss of batch to history
            train_history.append(loss.item())

        test_mae = 0

        #evaluation loop 
        for i, (x, y) in enumerate(test_loader):
            #cast to device
            x = x.to(device)
            y = y.to(device)

            #forward pass
            output = model(x)
            #calculate mae
            mae_error = mae(output, y)
            #add mae
            test_mae += mae_error.item()
        #compute average mae for entire test set
        test_mae /= len(test_loader)
        #append mae to history
        mae_history.append(test_mae)


        weight_threshold = 1e-5
        bias_threshold = 1e-5

        #prune all weights with absolute value less than weight_threshold
        for name, param in model.named_parameters():
            if 'weight' in name:
                param.data = torch.where(torch.abs(param.data) < weight_threshold, torch.zeros_like(param.data), param.data)

        #prune all biases with absolute value less than bias_threshold
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.data = torch.where(torch.abs(param.data) < bias_threshold, torch.zeros_like(param.data), param.data)


    #return model and history
    return model, train_history, mae_history