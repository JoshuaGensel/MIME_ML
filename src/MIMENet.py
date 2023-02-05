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


#training function
def train(training_path, epochs, learning_rate, batch_size, lambda_l1, hidden_size_factor, bottleneck, kdPath = None):

    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print device
    print("Device: " + str(device))

    #custom train loader
    train_loader = torch.utils.data.DataLoader(dataset=CustomTrainSet(training_path), batch_size=batch_size, shuffle=True)

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

    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #training history
    train_history = []

    #prediction history
    prediction_history = []

    #correlation history
    correlation_history_probs = []
    correlation_history_kds = []

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

        #prediction
        prediction_history.append(inferSingleProbabilities(model, input_size))

        #get kds
        if kdPath is not None:
            # read in kd values
            kds = np.loadtxt(kdPath)
            #insert 1 at position 0 and then every 3rd position
            kds = np.insert(kds, 0, 1)
            kds = np.insert(kds, np.arange(4, len(kds), 3), 1)
            #calculate correlation
            correlation_history_probs.append(np.corrcoef(prediction_history[-1], -np.log(kds))[0, 1])
            correlation_history_kds.append(np.corrcoef(1/np.array(prediction_history[-1])-1, kds)[0, 1])


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

    #define history dictionary
    history = {
        'training': train_history,
        'prediction': prediction_history
    }

    #if kd path is given, add kd correlation to history
    if kdPath is not None:
        history['correlation_probs'] = correlation_history_probs
        history['correlation_kds'] = correlation_history_kds

    #return model and history
    return model, history


def inferSingleProbabilities(model, numberFeatures):
    """
    Predicts binding probabilities for all possible mutations of a single position of the RNA sequence. For this, the model predicts
    synthetic data where only one mutation is present at a time. Ther order of the predictions is the following: position 1 wildtype,
    position 1 mutation 1, position 1 mutation 2, position 1 mutation 3, position 2 wildtype, position 2 mutation 1, etc.). The
    predictions are returned as a list of binding probabilities.

    Args:
        model (MIMENet.model): Model returned by train function of MIMENet used for prediction
        numberFeatures (int): Number of features in the dataset. This is 4 times the number of postitions of the RNA (from one 
        hot encoding) plus the number of protein concentrations times the number of rounds performed (8 for the classical MIME 
        experiment).

    Returns:
        list: a list of binding probabilities for each mutation along the RNA sequence
    """
    predictions = []
    prediction_example = np.zeros(numberFeatures)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in tqdm(range(8, numberFeatures)):
        current_prediction_example = prediction_example.copy()
        current_prediction_example[i] = 1
        current_prediction_example = torch.from_numpy(current_prediction_example).float()
        current_prediction_example = current_prediction_example.to(device)
        #output binding probability and append list for given protein concentration combination
        with torch.no_grad():
            output = model(current_prediction_example)
            predictions.append(output.item())

    return predictions

def inferPairwiseProbabilities(model, numberFeatures):
    """
    Predicts binding probabilities for all possible mutations of a pair of positions of the RNA sequence. For this, the model predicts
    synthetic data where two mutations are present at a time. Ther order of iterations through the possible mutations is the following:
    first position, second position, first mutation, second mutation. The predictions are returned as a list of binding probabilities.

    Args:
        model (MIMENet.model): Model returned by train function of MIMENet used for prediction
        numberFeatures (int): Number of features in the dataset. This is 4 times the number of postitions of the RNA (from one 
        hot encoding) plus the number of protein concentrations times the number of rounds performed (8 for the classical MIME 
        experiment).

    Returns:
        list: a list of binding probabilities for each mutation along the RNA sequence
    """
    
    predictionsPairwise = []
    prediction_example = np.zeros(numberFeatures)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in tqdm(range(8, numberFeatures,4)):
        for j in range(i+4, numberFeatures,4):
            for k in range(1,4):
                for l in range(1,4):
                        current_prediction_example = prediction_example.copy()
                        current_prediction_example[i+k] = 1
                        current_prediction_example[j+l] = 1
                        current_prediction_example = torch.from_numpy(current_prediction_example).float()
                        current_prediction_example = current_prediction_example.to(device)
                        #output binding probability and append list for given protein concentration combination
                        with torch.no_grad():
                            output = model(current_prediction_example)
                            predictionsPairwise.append(output.item())

    return predictionsPairwise

def inferEpistasis(model, numberFeatures, singleKdCoefficient, pairKdCoefficient, singleKdIntercept, pairKdIntercept):
    """
    Predicts epistasis values for all possible mutations of a pair of positions of the RNA sequence. For this, the model predicts
    synthetic data where two mutations are present at a time and compares them to the predictions of the single mutations individually. 
    The order of iterations through the possible mutations is the following: first position, second position, first mutation, second 
    mutation. The epistasis is calculated via the formula: ((output_pos1-0.5)*(output_pos2-0.5))/(output_pair-0.5).

    Args:
        model (MIMENet.model): Model returned by train function of MIMENet used for prediction
        numberFeatures (int): Number of features in the dataset. This is 4 times the number of postitions of the RNA (from one 
        hot encoding) plus the number of protein concentrations times the number of rounds performed (4*2=8 for the classical MIME 
        experiment).

    Returns:
        list: a list of binding probabilities for each mutation along the RNA sequence
    """

    epistasisPairwise = []
    singlePredictions = np.array(inferSingleProbabilities(model, numberFeatures))
    pairwisePredictions = np.array(inferPairwiseProbabilities(model, numberFeatures))
    singleKds = 1/singlePredictions+1
    pairwiseKds = 1/pairwisePredictions+1
    singleKdsLogScaled = 2**(singleKdCoefficient*np.log2(singleKds)+singleKdIntercept)
    pairwiseKdsLogScaled = 2**(pairKdCoefficient*np.log2(pairwiseKds)+pairKdIntercept)

    positionPairwise = 0
    for i in range(8, numberFeatures,4):
        for j in range(i+4, numberFeatures,4):
            for k in range(1,4):
                for l in range(1,4):
                    pos1 = singleKdsLogScaled[i+k-8]
                    pos2 = singleKdsLogScaled[j+l-8]
                    pair = pairwiseKdsLogScaled[positionPairwise]
                    positionPairwise += 1
                    epistasisPairwise.append(((pos1)*(pos2))/(pair))


    return epistasisPairwise