import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import exists

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
        # set file path
        self.path = path
        # read first line as text
        with open(self.path) as f:
            self.first_line = f.readline()
        # split first line by underscore
        self.training_length = len(self.first_line.split('_')[0])
        # get number of lines in file
        self.len = sum(1 for line in open(self.path))
        print("Number of training examples: " + str(self.len))


    def __getitem__(self, index):
        # np.genfromtxt at index line
        x = np.genfromtxt(self.path, delimiter=1, dtype=int, skip_header=index, max_rows=1, usecols=range(self.training_length))
        y = np.genfromtxt(self.path, delimiter=1, dtype=int, skip_header=index, max_rows=1, usecols=self.training_length+1)
        # convert to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y

    def __len__(self):
        return self.len


#training function
def train(training_path, epochs, learning_rate, batch_size, lambda_l2, hidden_size_factor, bottleneck, kd_path = None, model_path = None, backup_interval = 10):

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

    epoch = 0

    # if model file already exists, load it
    if exists(model_path + ".pt"):
        model.load_state_dict(torch.load(model_path))
        print("Loaded model from " + str(model_path))
        #load training history
        train_history = np.load(model_path + "_train_history.npy").tolist()
        #load prediction history
        prediction_history = np.load(model_path + "_prediction_history.npy").tolist()
        #load correlation history
        correlation_history_probs = np.load(model_path + "_correlation_history_probs.npy").tolist()
        correlation_history_kds = np.load(model_path + "_correlation_history_kds.npy").tolist()
        #load epoch
        epoch = np.load(model_path + "_epoch.npy").tolist()
        #load optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)
        optimizer.load_state_dict(torch.load(model_path + "_optimizer.pt"))
        print("Loaded optimizer from " + str(model_path + "_optimizer.pt"))

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
    for epoch in tqdm(range(epochs - epoch)):

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

        # if epoch is multiple of backup interval, save model
        if epoch % backup_interval == 0:
            #save model
            torch.save(model.state_dict(), model_path + ".pt")
            #save history
            np.savetxt(model_path + "_train_history.txt", train_history)
            np.savetxt(model_path + "_prediction_history.txt", prediction_history)
            if kd_path is not None:
                np.savetxt(model_path + "_correlation_history_probs.txt", correlation_history_probs)
                np.savetxt(model_path + "_correlation_history_kds.txt", correlation_history_kds)
            np.savetxt(model_path + "_epoch.txt", np.array([epoch]))
            torch.save(optimizer.state_dict(), model_path + "_optimizer.pt")



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

def inferSingleKdsProtein(model, numberFeatures, n : int, number_protein_concentrations : int):
    
    predictedKds = dict()
    for protein_concentration_1 in range(number_protein_concentrations):
        for protein_concentration_2 in range(number_protein_concentrations):
            predictions = []
            prediction_example = np.zeros(numberFeatures)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            for pos in tqdm(range(6, numberFeatures, 4)):
                for mut in range(1,4):
                    wildtype_prediction_example = prediction_example.copy()
                    mutation_prediction_example = prediction_example.copy()
                    wildtype_prediction_example[protein_concentration_1] = 1
                    wildtype_prediction_example[protein_concentration_2+number_protein_concentrations] = 1
                    wildtype_prediction_example[pos] = 1
                    mutation_prediction_example[pos+mut] = 1
                    mutation_prediction_example[protein_concentration_1] = 1
                    mutation_prediction_example[protein_concentration_2+number_protein_concentrations] = 1
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
                        predictions.append(correctedKd.tolist())
            predictedKds[(protein_concentration_1, protein_concentration_2)] = predictions

    return predictedKds


def inferEpistasis(singleKds : list, pairwiseKds : list):
    
    single_kd_pred = np.array(singleKds)
    single_kd_pred_means = np.mean(single_kd_pred, axis=1)
    single_kd_pred_conf = np.zeros((single_kd_pred_means.shape[0], 2))
    for i in range(single_kd_pred_means.shape[0]):
        single_kd_pred_conf[i] = np.quantile(single_kd_pred[i], [0.025, 0.975])

    pairwise_kd_pred = np.array(pairwiseKds)
    pairwise_kd_pred_means = np.mean(pairwise_kd_pred, axis=1)
    pairwise_kd_pred_conf = np.zeros((pairwise_kd_pred_means.shape[0], 2))
    for i in range(pairwise_kd_pred_means.shape[0]):
        pairwise_kd_pred_conf[i] = np.quantile(pairwise_kd_pred[i], [0.025, 0.975])

    #iterate through all possible pairs
    epistasis = []
    i = 0
    for pos1 in range(single_kd_pred.shape[0], 3):
        for pos2 in range(pos1+1, single_kd_pred.shape[0], 3):
            for mut1 in range(3):
                for mut2 in range(3):
            
                    # check if the lower bounds of the intervals are both above 1
                    single_kds_above_1 = single_kd_pred_conf[pos1+mut1][0] > 1 and single_kd_pred_conf[pos2+mut2][0] > 1

                    # check if the upper bounds of the intervals are both below 1
                    single_kds_below_1 = single_kd_pred_conf[pos1+mut1][1] < 1 and single_kd_pred_conf[pos2+mut2][1] < 1

                    # check if pairwise interval contains 1
                    pairwise_kd_contains_1 = pairwise_kd_pred_conf[i][0] < 1 and pairwise_kd_pred_conf[i][1] > 1

                    # check if the lower bound of the pairwise interval is above 1
                    pairwise_kd_above_1 = pairwise_kd_pred_conf[i][0] > 1

                    # check if the upper bound of the pairwise interval is below 1
                    pairwise_kd_below_1 = pairwise_kd_pred_conf[i][1] < 1

                    if (single_kds_above_1 and pairwise_kd_contains_1) or (single_kds_below_1 and pairwise_kd_contains_1) or (single_kds_above_1 and pairwise_kd_below_1) or (single_kds_below_1 and pairwise_kd_above_1):
                        epistasis.append(1)
                    else:
                        epistasis.append(0)
                    i += 1

    return epistasis

