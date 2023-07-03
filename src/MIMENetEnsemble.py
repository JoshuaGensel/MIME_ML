import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import exists
import json

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
        # set path
        self.path = path
        # get number of lines at path
        self.len = sum(1 for line in open(path))
        # get line offsets
        self.line_offsets = []
        with open(path) as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line)
            f.seek(0)
        #print number of training examples        
        print("Number of Training examples: " + str(self.len))


    def __getitem__(self, index):
        # get line at index with seek
        with open(self.path) as f:
            f.seek(self.line_offsets[index])
            line = f.readline()
        #split line
        line = line.split('_')
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

    epoch = 0

    # if model file already exists, load it
    if exists(model_path + ".pt"):
        model.load_state_dict(torch.load(model_path+".pt"))
        print("Loaded model from " + str(model_path+".pt"))
        #load train_history.txt
        train_history = np.loadtxt(model_path + "_train_history.txt").tolist()        
        #load prediction history
        prediction_history = np.loadtxt(model_path + "_prediction_history.txt").tolist()
        #load correlation history
        if kd_path is not None:
            correlation_history_probs = np.loadtxt(model_path + "_correlation_history_probs.txt").tolist()
            correlation_history_kds = np.loadtxt(model_path + "_correlation_history_kds.txt").tolist()
        #load epoch
        epoch = np.loadtxt(model_path + "_epoch.txt", dtype=int)
        #load optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)
        optimizer.load_state_dict(torch.load(model_path + "_optimizer.pt"))
        print("Loaded optimizer from " + str(model_path + "_optimizer.pt"))

    #training loop
    for epo in range(epoch, epochs):

        #  print("Epoch: " + str(epoch+1) + "/" + str(epochs))

        #training loop
        for x, y in tqdm(train_loader, desc="Epoch " + str(epo+1) + "/" + str(epochs), leave=False):
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
        if epo % backup_interval == 0:
            #save model
            torch.save(model.state_dict(), model_path + ".pt")
            #save history
            np.savetxt(model_path + "_train_history.txt", train_history)
            np.savetxt(model_path + "_prediction_history.txt", prediction_history)
            if kd_path is not None:
                np.savetxt(model_path + "_correlation_history_probs.txt", correlation_history_probs)
                np.savetxt(model_path + "_correlation_history_kds.txt", correlation_history_kds)
            np.savetxt(model_path + "_epoch.txt", np.array([epo]).astype(int))
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

    # save model
    torch.save(model.state_dict(), model_path + "_final_model.pt")
    # save history as json
    with open(model_path + "_history.json", 'w') as fp:
        json.dump(history, fp)

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
                            prediction = model.predict(current_prediction_example, n)
                            predictionsPairwise.append(prediction)

    return predictionsPairwise

def inferSingleKds(model, n_protein_concentrations, n_rounds, path_wildtype, n : int):

    # read in wildtype
    with open(path_wildtype, 'r') as f:
        wildtype = f.read()

    n_features = len(wildtype) * 4 + n_protein_concentrations * n_rounds
    
    kds_nucleotide = []
    kds_position = []
    prediction_example = np.zeros(n_features)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for pos, feature in enumerate(tqdm(range(n_protein_concentrations*n_rounds, n_features, 4))):
        
        prediction_example_wildtype = prediction_example.copy()
        prediction_example_mut1 = prediction_example.copy()
        prediction_example_mut2 = prediction_example.copy()
        prediction_example_mut3 = prediction_example.copy()

        # get wildtype at position
        wildtype_nucleotide = wildtype[pos]

        order_nucleotides = ['A', 'C', 'G', 'T']

        # get index of wildtype nucleotide
        index_wildtype = order_nucleotides.index(wildtype_nucleotide)

        # set wildtype nucleotide
        prediction_example_wildtype[feature + index_wildtype] = 1

        # set mutant nucleotides
        indices_mutants = [i for i in range(4) if i != index_wildtype]
        prediction_example_mut1[feature + indices_mutants[0]] = 1
        prediction_example_mut2[feature + indices_mutants[1]] = 1
        prediction_example_mut3[feature + indices_mutants[2]] = 1


        # convert all to tensors
        prediction_example_wildtype = torch.tensor(prediction_example_wildtype, dtype=torch.float32).to(device)
        prediction_example_mut1 = torch.tensor(prediction_example_mut1, dtype=torch.float32).to(device)
        prediction_example_mut2 = torch.tensor(prediction_example_mut2, dtype=torch.float32).to(device)
        prediction_example_mut3 = torch.tensor(prediction_example_mut3, dtype=torch.float32).to(device)

        # predict
        with torch.no_grad():
            wildtype_prediction = model.predict(prediction_example_wildtype, n)
            mut1_prediction = model.predict(prediction_example_mut1, n)
            mut2_prediction = model.predict(prediction_example_mut2, n)
            mut3_prediction = model.predict(prediction_example_mut3, n)

            # calculate kd
            wildtype_kd = 1 / np.array(wildtype_prediction) - 1
            mut1_kd = 1 / np.array(mut1_prediction) - 1
            mut2_kd = 1 / np.array(mut2_prediction) - 1
            mut3_kd = 1 / np.array(mut3_prediction) - 1

            # correct kds
            mut1_kd_rel = mut1_kd / wildtype_kd
            mut2_kd_rel = mut2_kd / wildtype_kd
            mut3_kd_rel = mut3_kd / wildtype_kd

            # average kds over the 3 mutations
            pos_kd = np.mean([mut1_kd_rel, mut2_kd_rel, mut3_kd_rel], axis=0)
            
            # append to lists
            kds_nucleotide.append(mut1_kd_rel.tolist())
            kds_nucleotide.append(mut2_kd_rel.tolist())
            kds_nucleotide.append(mut3_kd_rel.tolist())
            kds_position.append(pos_kd.tolist())

    return kds_nucleotide, kds_position

def inferPairwiseKds(model, n_protein_concentrations, n_rounds, path_wildtype, n : int):

    # read in wildtype
    with open(path_wildtype, 'r') as f:
        wildtype = f.read()

    n_features = len(wildtype) * 4 + n_protein_concentrations * n_rounds
    
    kds_pairwise = []
    prediction_example = np.zeros(n_features)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for pos1, feature1 in enumerate(range(n_protein_concentrations*n_rounds, n_features, 4)):
        for pos2, feature2 in enumerate(tqdm(range(feature1+4, n_features, 4), desc='Position ' + str(pos1+1) + '/' + str(len(wildtype)) + ' ', leave=False)):
            
            # get wildtype at both positions
            wildtype1 = wildtype[pos1]
            wildtype2 = wildtype[pos2]

            order_nucleotides = ['A', 'C', 'G', 'T']

            index_wildtype1 = order_nucleotides.index(wildtype1)
            index_wildtype2 = order_nucleotides.index(wildtype2)

            # iterate over all possible mutations
            for mut1 in range(4):
                for mut2 in range(4):

                    # skip wildtype
                    if mut1 == index_wildtype1 or mut2 == index_wildtype2:
                        continue

                    prediction_example_wildtype = prediction_example.copy()
                    prediction_example_mutant = prediction_example.copy()

                    # set wildtype
                    prediction_example_wildtype[feature1+index_wildtype1] = 1
                    prediction_example_wildtype[feature2+index_wildtype2] = 1

                    # set mutant
                    prediction_example_mutant[feature1+mut1] = 1
                    prediction_example_mutant[feature2+mut2] = 1

                    # print('wildtypes: ', feature1+index_wildtype1, feature2+index_wildtype2)
                    # print('mutants: ', feature1+mut1, feature2+mut2)

                    # convert to tensor
                    prediction_example_wildtype = torch.tensor(prediction_example_wildtype, dtype=torch.float32).to(device)
                    prediction_example_mutant = torch.tensor(prediction_example_mutant, dtype=torch.float32).to(device)

                    # predict
                    with torch.no_grad():
                        prediction_wildtype = model.predict(prediction_example_wildtype, n)
                        prediction_mutant = model.predict(prediction_example_mutant, n)

                    # calculate kd
                    wildtype_kd = 1 / np.array(prediction_wildtype) - 1
                    mutant_kd = 1 / np.array(prediction_mutant) - 1

                    # calculate relative kd
                    relative_kd = mutant_kd / wildtype_kd

                    # append to list
                    kds_pairwise.append(relative_kd.tolist())

    return kds_pairwise

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
    pairs = []
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
                    pairs.append((pos1+mut1, pos2+mut2))

    return epistasis, pairs

def computeEpistasis(singleKds : list, pairwiseKds : list):
    
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
    epistasis_nucleotides = []
    epistasis_positions = []
    nucleotide_pairs = []
    position_pairs = []
    j = 0
    for pos1 in range(0,single_kd_pred.shape[0], 3):
        for pos2 in range(pos1+3, single_kd_pred.shape[0], 3):
            for mut1 in range(3):
                for mut2 in range(3):
                    epistasis_nucleotides.append((single_kd_pred_means[pos1+mut1]*single_kd_pred_means[pos2+mut2])/pairwise_kd_pred_means[j])
                    j += 1
                    nucleotide_pairs.append((pos1+mut1, pos2+mut2))
            
            # compute average epistasis for each mutation per pos 1 and pos 2
            epistasis_positions.append(np.mean(epistasis_nucleotides[-9:]))
            position_pairs.append((pos1//3, pos2//3))
            

    return epistasis_nucleotides, nucleotide_pairs, epistasis_positions, position_pairs
