import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm

# MIMENet Deep Neural Network
class MIMENetEnsemble(nn.Module):
    # Constructor for 5 fully connected layers with bottleneck for layers 4 and 5
    # dropout after each layer
    def __init__(self, input_size, hidden_size_factor, bottleneck, output_size):
        super(MIMENetEnsemble, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_factor*input_size)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_size_factor*input_size, hidden_size_factor*input_size)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(hidden_size_factor*input_size, hidden_size_factor*input_size)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(hidden_size_factor*input_size, int(hidden_size_factor*input_size*bottleneck))
        self.dropout4 = nn.Dropout(p=0.2)
        self.fc5 = nn.Linear(int(hidden_size_factor*input_size*bottleneck), output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # Forward pass
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.sigmoid(self.fc5(x))

        return x
    
    def predict(self, x, n):
        # forward pass n times with dropout disabled
        predictions = []
        for i in range(n):
            # appen output as float
            predictions.append(self.forward(x).item())
        return predictions
    
        

#custom training set
class CustomTrainingSet(torch.utils.data.Dataset):
    #read data from text file
    #training examples and labels are separated by an underscore
    #training examples are not separated
    def __init__(self, path):
        #read file
        file = open(path, 'r')
        #read lines
        self.lines = file.readlines()
        #close file
        file.close()
        #get length
        self.len = len(self.lines)
        print("Number of training examples: " + str(self.len))

    def __getitem__(self, index):
        #split line
        line = self.lines[index].split('_')
        #append training example to x
        # inputs are not separated by spaces
        x = [float(i) for i in line[0]]
        #append label to y
        y = [float(line[1])]
        #convert to torch tensors
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y

    def __len__(self):
        return self.len


#training function
def train(training_path, epochs, learning_rate, batch_size, lambda_l2, hidden_size_factor, bottleneck, kd_path = None, model_path = None):

    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print device
    print("Device: " + str(device))

    #custom train loader
    train_loader = torch.utils.data.DataLoader(dataset=CustomTrainingSet(training_path), batch_size=batch_size, shuffle=True)

    #get input size
    input_size = len(open(training_path).readline().split('_')[0])

    #get output size
    output_size = 1

    #initialize model
    model = MIMENetEnsemble(input_size, hidden_size_factor, bottleneck, output_size)

    # load model if path is given
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    #move model to device
    model.to(device)

    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)

    #training history
    train_history = []

    #prediction history
    prediction_history = []

    #correlation history
    correlation_history_probs = []
    correlation_history_kds = []

    #training loop
    for epoch in tqdm(range(epochs)):

        #  print("Epoch: " + str(epoch+1) + "/" + str(epochs))

        #training loop
        for i, (x, y) in enumerate(train_loader):
            #cast to device
            x = x.to(device)
            y = y.to(device)

            #forward pass
            output = model(x)
            #calculate binary cross entropy loss
            loss = F.binary_cross_entropy(output, y)
            #clear gradients
            optimizer.zero_grad()
            #backward pass
            loss.backward()
            #update weights
            optimizer.step()
            #append loss of batch to history
            train_history.append(loss.item())

        #prediction
        prediction_array = np.array(inferSingleProbabilities(model, input_size, 100))
        # rows are predictions for one position
        # take mean of each row
        predictions = np.mean(prediction_array, axis=1)
        #append to prediction history as list
        prediction_history.append(predictions.tolist())

        #get kds
        if kd_path is not None:
            # read in kd values
            kds = np.loadtxt(kd_path)
            #insert 1 at position 0 and then every 3rd position
            kds = np.insert(kds, 0, 1)
            kds = np.insert(kds, np.arange(4, len(kds), 3), 1)
            #calculate correlation
            correlation_history_probs.append(np.corrcoef(prediction_history[-1], -np.log(kds))[0, 1])
            correlation_history_kds.append(np.corrcoef(1/np.array(prediction_history[-1])-1, kds)[0, 1])


    #define history dictionary
    history = {
        'training': train_history,
        'prediction': prediction_history
    }

    #if kd path is given, add kd correlation to history
    if kd_path is not None:
        history['correlation_probs'] = correlation_history_probs
        history['correlation_kds'] = correlation_history_kds

    #return model and history
    return model, history

def inferSingleProbabilities(model, numberFeatures, n : int):
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
    for i in tqdm(range(6, numberFeatures), leave=False):
        current_prediction_example = prediction_example.copy()
        current_prediction_example[i] = 1
        current_prediction_example = torch.from_numpy(current_prediction_example).float()
        current_prediction_example = current_prediction_example.to(device)
        with torch.no_grad():
            prediction = model.predict(current_prediction_example, n)
            predictions.append(prediction)

    return predictions

def inferPairwiseProbabilities(model, numberFeatures, n : int):
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
    for i in tqdm(range(6, numberFeatures,4)):
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
                            prediction = model.predict(current_prediction_example, n)
                            predictionsPairwise.append(prediction)

    return predictionsPairwise

def inferSingleKds(model, numberFeatures, n : int):
    """
    Predicts Kd values for all possible mutations of a single position of the RNA sequence. For this, the model predicts
    synthetic data where only one mutation is present at a time. The order of the predictions is the following: position 1 mutation 1, 
    position 1 mutation 2, position 1 mutation 3, position 2 mutation 1, etc.). The predictions are returned as a list of Kd values.

    Args:
        model (MIMENet.model): Model returned by train function of MIMENet used for prediction
        numberFeatures (int): Number of features in the dataset. This is 4 times the number of postitions of the RNA (from one 
        hot encoding) plus the number of protein concentrations times the number of rounds performed (8 for the classical MIME 
        experiment).

    Returns:
        list: a list of Kd values for each mutation along the RNA sequence
    """
    predictedKds = []
    prediction_example = np.zeros(numberFeatures)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for pos in tqdm(range(6, numberFeatures, 4)):
        for mut in range(1,4):
            wildtype_prediction_example = prediction_example.copy()
            mutation_prediction_example = prediction_example.copy()
            wildtype_prediction_example[pos] = 1
            mutation_prediction_example[pos+mut] = 1
            wildtype_prediction_example = torch.from_numpy(wildtype_prediction_example).float()
            mutation_prediction_example = torch.from_numpy(mutation_prediction_example).float()
            wildtype_prediction_example = wildtype_prediction_example.to(device)
            mutation_prediction_example = mutation_prediction_example.to(device)
            with torch.no_grad():
                wildtype_output = model.predict(wildtype_prediction_example, n)
                mutation_output = model.predict(mutation_prediction_example, n)
                #calculate Kds
                wildtypeKd = 1/np.array(wildtype_output) - 1
                mutationKd = 1/np.array(mutation_output) - 1
                correctedKd = mutationKd / wildtypeKd
                predictedKds.append(correctedKd.tolist())

    return predictedKds

def inferPairwiseKds(model, numberFeatures, n : int):
    """
    Predicts Kd values for all possible mutations of a pair of positions of the RNA sequence. For this, the model predicts
    synthetic data where two mutations are present at a time. The order of iterations through the possible mutations is the following:
    first position, second position, first mutation, second mutation. The predictions are returned as a list of Kd values.

    Args:
        model (MIMENet.model): Model returned by train function of MIMENet used for prediction
        numberFeatures (int): Number of features in the dataset. This is 4 times the number of postitions of the RNA (from one 
        hot encoding) plus the number of protein concentrations times the number of rounds performed (8 for the classical MIME 
        experiment).

    Returns:
        list: a list of Kd values for each mutation along the RNA sequence
    """
    predictedPairwiseKds = []
    prediction_example = np.zeros(numberFeatures)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for pos1 in tqdm(range(6, numberFeatures,4)):
        for pos2 in range(pos1+4, numberFeatures,4):
            for mut1 in range(1,4):
                for mut2 in range(1,4):
                        wildtype_prediction_example = prediction_example.copy()
                        mutation_prediction_example = prediction_example.copy()
                        wildtype_prediction_example[pos1] = 1
                        wildtype_prediction_example[pos2] = 1
                        mutation_prediction_example[pos1+mut1] = 1
                        mutation_prediction_example[pos2+mut2] = 1
                        wildtype_prediction_example = torch.from_numpy(wildtype_prediction_example).float()
                        mutation_prediction_example = torch.from_numpy(mutation_prediction_example).float()
                        wildtype_prediction_example = wildtype_prediction_example.to(device)
                        mutation_prediction_example = mutation_prediction_example.to(device)
                        #get output binding probabilities
                        with torch.no_grad():
                            wildtype_prediciton = model.predict(wildtype_prediction_example, n)
                            mutation_prediciton = model.predict(mutation_prediction_example, n)
                            #calculate Kd and append list
                            wildtypeKd = 1/np.array(wildtype_prediciton) - 1
                            mutationKd = 1/np.array(mutation_prediciton) - 1
                            correctedKd = mutationKd / wildtypeKd
                            predictedPairwiseKds.append(correctedKd.tolist())

    return predictedPairwiseKds
