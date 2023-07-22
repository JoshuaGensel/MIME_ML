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
    
    
        

#custom training set
class custom_data_set(torch.utils.data.Dataset):
    #read data from text file
    #training examples and labels are separated by an underscore
    #training examples are not separated
    def __init__(self, path, name = "training"):
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
        print(f"Number of {name} examples: " + str(self.len))


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

class inference_dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # convert from numpy to tensor
        x = self.data[index]

        return x

#training function
def train(training_path, test_path, epochs, learning_rate, batch_size, lambda_l2, hidden_size_factor, bottleneck, kd_path = None, model_path = None, backup_interval = 10):

    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print device
    print("Device: " + str(device))

    #custom train loader
    train_loader = torch.utils.data.DataLoader(dataset=custom_data_set(training_path), batch_size=batch_size, shuffle=True)

    #custom test loader
    test_loader = torch.utils.data.DataLoader(dataset=custom_data_set(test_path, "testing"), batch_size=batch_size, shuffle=False)

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

    #test history
    test_history = []

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
        #load test_history.txt
        test_history = np.loadtxt(model_path + "_test_history.txt").tolist()
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

        #test loop
        loss_epoch = 0
        with torch.no_grad():
            #set model to eval mode
            model.eval()
            for x, y in tqdm(test_loader, desc="Testing - Epoch " + str(epo+1) + "/" + str(epochs), leave=False):
                #cast to device
                x = x.to(device)
                y = y.to(device)

                #forward pass
                output = model(x)
                #calculate binary cross entropy loss
                loss = F.binary_cross_entropy(output, y)
                #add loss to epoch loss
                loss_epoch += loss.item()
            #set model to train mode
            model.train()
        #append loss of epoch to history
        test_history.append(loss_epoch/len(test_loader))

        #prediction
        prediction_history.append(inferSingleProbabilities(model, input_size, batch_size))

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
            torch.save(optimizer.state_dict(), model_path + "_optimizer.pt")
            #backup model at current epoch
            torch.save(model.state_dict(), model_path + "_epoch_" + str(epo+1) + ".pt")
            #save history
            np.savetxt(model_path + "_train_history.txt", train_history)
            np.savetxt(model_path + "_test_history.txt", test_history)
            np.savetxt(model_path + "_prediction_history.txt", prediction_history)
            if kd_path is not None:
                np.savetxt(model_path + "_correlation_history_probs.txt", correlation_history_probs)
                np.savetxt(model_path + "_correlation_history_kds.txt", correlation_history_kds)
            np.savetxt(model_path + "_epoch.txt", np.array([epo]).astype(int)+1)
            



    #define history dictionary
    history = {
        'training': train_history,
        'test': test_history,
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

def predict(model, x, n, batch_size):
    """This function is used for inference with the model. It takes in a 2d tensor
    and predicts the output using the model. For uncertainty quantification, the
    model is run n times and the full distribution is returned.

    Args:
        model (MIMENet.model): Model returned by train function of MIMENet used for prediction
        x (tensor): Input tensor
        n (int): Number of times to run the model
    """
    # set length of input
    length = x.shape[0]

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: " + str(device))
    
    # define x as inference dataset
    x = inference_dataset(x)

    # define dataloader
    dataloader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=False)

    # set model
    model = model.to(device)

    # initialize output
    output = np.zeros((length, n))
    # run model n times
    for i in tqdm(range(n)):
        for j, data in enumerate(dataloader):
            # only send data to device
            data = data.to(device)

            # add to output
            output[j*batch_size:(j+1)*batch_size, i] = model(data).squeeze().cpu().detach().numpy()
            
    return output

def output(model, x, batch_size):
    """This function is used for inference with the model. It takes in a 2d tensor
    and predicts the output using the model. Instead of running the model multiple
    times, it turns of the models dropout and returns the output of the model.

    Args:
        model (MIMENet.model): Model returned by train function of MIMENet used for prediction
        x (tensor): Input tensor
        batch_size (int): Batch size for inference
    """

    # set length of input
    length = x.shape[0]

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # define x as inference dataset
    x = inference_dataset(x)

    # define dataloader
    dataloader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=False)

    # set model
    model = model.to(device)

    # set model to eval mode
    model.eval()

    # initialize output
    output = np.zeros(length)
    # run model
    for j, data in enumerate(dataloader):
        # only send data to device
        data = data.to(device)

        # add to output
        output[j*batch_size:(j+1)*batch_size] = model(data).squeeze().cpu().detach().numpy()

    # set model to train mode
    model.train()
            
    return output

def inferSingleProbabilities(model, numberFeatures, batch_size : int):
    """
    Predicts binding probabilities for all input feature for the model. For this, the model predicts
    synthetic data where only one feature is present at a time. Ther order of the predictions is the
    following: round 1 protein concenctration 1, round 1 protein concentration 2, round 2 protein 
    concentration 1, etc., position 1 wildtype, position 1 mutation 1, position 1 mutation 2, 
    position 1 mutation 3, position 2 wildtype, position 2 mutation 1, etc.). The predictions are 
    returned as a list of probabilities.

    Args:
        model (MIMENet.model): Model returned by train function of MIMENet used for prediction
        numberFeatures (int): Number of features in the dataset. This is 4 times the number of postitions of the RNA (from one 
        hot encoding) plus the number of protein concentrations times the number of rounds performed (8 for the classical MIME 
        experiment).

    Returns:
        list: a list of binding probabilities for each mutation along the RNA sequence
    """
    
    # initialize matrix of input features x input features
    prediction_example = np.zeros((numberFeatures, numberFeatures))
    # set diagonal to 1
    np.fill_diagonal(prediction_example, 1)
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model output of prediction example
    with torch.no_grad():
        prediction = output(model, torch.from_numpy(prediction_example).float().to(device), batch_size)
    # return prediction as list
    return prediction.tolist()    


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
                            prediction = predict(current_prediction_example, n)
                            predictionsPairwise.append(prediction)

    return predictionsPairwise

def inferSingleKds(model, n_protein_concentrations, n_rounds, path_wildtype, n : int, batch_size : int):

    # read in wildtype
    with open(path_wildtype, 'r') as f:
        wildtype = f.read()

    n_features = len(wildtype) * 4 + n_protein_concentrations * n_rounds
    
    # initialize prediction example (number of nucleotides by number of features)
    prediction_example_mutation = np.zeros((len(wildtype)*3, n_features))
    prediction_example_wildtype = np.zeros((len(wildtype), n_features))

    for pos, feature in enumerate(range(n_protein_concentrations*n_rounds, n_features, 4)):

        # get wildtype at position
        wildtype_nucleotide = wildtype[pos]

        order_nucleotides = ['A', 'C', 'G', 'T']

        # get index of wildtype nucleotide
        index_wildtype = order_nucleotides.index(wildtype_nucleotide)

        # set wildtype nucleotide
        prediction_example_wildtype[pos, feature + index_wildtype] = 1

        # set mutant nucleotides
        indices_mutants = [i for i in range(4) if i != index_wildtype]
        prediction_example_mutation[pos*3, feature + indices_mutants[0]] = 1
        prediction_example_mutation[pos*3+1, feature + indices_mutants[1]] = 1
        prediction_example_mutation[pos*3+2, feature + indices_mutants[2]] = 1


    # predict
    with torch.no_grad():
        wildtype_prediction = predict(model, prediction_example_wildtype, n, batch_size)
        mutation_prediction = predict(model, prediction_example_mutation, n, batch_size)

        # compute predictions to kds
        wildtype_prediction = 1 / wildtype_prediction - 1
        mutation_prediction = 1 / mutation_prediction - 1

        # repeat every row in wildtype_prediction 3 times
        wildtype_prediction = np.repeat(wildtype_prediction, 3, axis=0)

        # correct kds
        kds_nucleotide = mutation_prediction / wildtype_prediction

        # reshape to 3d so that 3 rows are grouped together as one position
        kds_position = kds_nucleotide.reshape((len(wildtype), 3, n))

        # take max of 3 rows
        kds_position = np.max(kds_position, axis=1)


    return kds_nucleotide, kds_position

def getSingleKds(model, n_protein_concentrations, n_rounds, path_wildtype, batch_size : int):

    # read in wildtype
    with open(path_wildtype, 'r') as f:
        wildtype = f.read()

    n_features = len(wildtype) * 4 + n_protein_concentrations * n_rounds
    
    # initialize prediction example (number of nucleotides by number of features)
    prediction_example_mutation = np.zeros((len(wildtype)*3, n_features))
    prediction_example_wildtype = np.zeros((len(wildtype), n_features))

    for pos, feature in enumerate(range(n_protein_concentrations*n_rounds, n_features, 4)):

        # get wildtype at position
        wildtype_nucleotide = wildtype[pos]

        order_nucleotides = ['A', 'C', 'G', 'T']

        # get index of wildtype nucleotide
        index_wildtype = order_nucleotides.index(wildtype_nucleotide)

        # set wildtype nucleotide
        prediction_example_wildtype[pos, feature + index_wildtype] = 1

        # set mutant nucleotides
        indices_mutants = [i for i in range(4) if i != index_wildtype]
        prediction_example_mutation[pos*3, feature + indices_mutants[0]] = 1
        prediction_example_mutation[pos*3+1, feature + indices_mutants[1]] = 1
        prediction_example_mutation[pos*3+2, feature + indices_mutants[2]] = 1


    # predict
    with torch.no_grad():
        wildtype_prediction = output(model, prediction_example_wildtype, batch_size)
        mutation_prediction = output(model, prediction_example_mutation, batch_size)

        # compute predictions to kds
        wildtype_prediction = 1 / wildtype_prediction - 1
        mutation_prediction = 1 / mutation_prediction - 1

        # repeat every row in wildtype_prediction 3 times
        wildtype_prediction = np.repeat(wildtype_prediction, 3, axis=0)

        # correct kds
        kds_nucleotide = mutation_prediction / wildtype_prediction

        # reshape to 3d so that 3 rows are grouped together as one position
        kds_position = kds_nucleotide.reshape((len(wildtype), 3))

        # take max of 3 rows
        kds_position = np.max(kds_position, axis=1)


    return kds_nucleotide, kds_position

def inferPairwiseKds(model, n_protein_concentrations, n_rounds, path_wildtype, n : int, batch_size : int):

    # read in wildtype
    with open(path_wildtype, 'r') as f:
        wildtype = f.read()

    n_features = len(wildtype) * 4 + n_protein_concentrations * n_rounds
    n_pairs_mut = int((len(wildtype)*(len(wildtype)-1)/2)*3*3)
    n_pairs_wt = int((len(wildtype)*(len(wildtype)-1)/2))
    
    # initialize prediction example
    prediction_example_mutation = np.zeros((n_pairs_mut, n_features))
    prediction_example_wildtype = np.zeros((n_pairs_wt, n_features))

    i = 0
    for pos1, feature1 in enumerate(range(n_protein_concentrations*n_rounds, n_features, 4)):
        for pos2, feature2 in enumerate(range(feature1+4, n_features, 4)):
            
            # get wildtype at both positions
            wildtype1 = wildtype[pos1]
            wildtype2 = wildtype[pos2]

            order_nucleotides = ['A', 'C', 'G', 'T']

            index_wildtype1 = order_nucleotides.index(wildtype1)
            index_wildtype2 = order_nucleotides.index(wildtype2)

            # set wildtype nucleotides
            prediction_example_wildtype[i//9, feature1 + index_wildtype1] = 1
            prediction_example_wildtype[i//9, feature2 + index_wildtype2] = 1

            # iterate over all possible mutations
            for mut1 in range(4):
                for mut2 in range(4):

                    # skip wildtype
                    if mut1 == index_wildtype1 or mut2 == index_wildtype2:
                        continue

                    # set mutant
                    prediction_example_mutation[i, feature1+mut1] = 1
                    prediction_example_mutation[i, feature2+mut2] = 1

                    i += 1

    # predict
    with torch.no_grad():

        wildtype_prediction = predict(model, prediction_example_wildtype, n, batch_size)
        mutation_prediction = predict(model, prediction_example_mutation, n, batch_size)

        # compute predictions to kds
        wildtype_prediction = 1 / wildtype_prediction - 1
        mutation_prediction = 1 / mutation_prediction - 1

        # repeat every row in wildtype_prediction 3 times
        wildtype_prediction = np.repeat(wildtype_prediction, 9, axis=0)

        # correct kds
        kds_pairwise = mutation_prediction / wildtype_prediction


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

    pairwise_kd_pred = np.array(pairwiseKds)

    num_tests = pairwise_kd_pred.shape[0]
    #iterate through all possible pairs
    epistasis_nuc = []
    pairs_nuc = []
    epistasis_pos = []
    pairs_pos = []
    i = 0
    for pos1 in range(0, single_kd_pred.shape[0], 3):
        for pos2 in range(pos1+3, single_kd_pred.shape[0], 3):
            for mut1 in range(3):
                for mut2 in range(3):
            
                    # get single kds
                    single_kd_1 = single_kd_pred[pos1+mut1]
                    single_kd_2 = single_kd_pred[pos2+mut2]
                    # get pairwise kd
                    pairwise_kd = pairwise_kd_pred[i]

                    # check how often the pairwise kd is lower than the product of the single kds
                    check = pairwise_kd < single_kd_1 * single_kd_2
                    count = np.sum(check)

                    # define threshold as 0.95 with bonferroni correction
                    threshold = 1 - (0.05 / num_tests)

                    # predict epistasis if more than threshold of the predictions are lower
                    if count / pairwise_kd.shape[0] > threshold:
                        epistasis_nuc.append(1)
                    else:
                        epistasis_nuc.append(0)

                    # epistasis.append(count / pairwise_kd.shape[0])

                    pairs_nuc.append((pos1+mut1, pos2+mut2))

                    i += 1

            # check if for all 9 pairs of mutations at the two positions, there is epistasis
            if np.sum(epistasis_nuc[-9:]) == 9:
                epistasis_pos.append(1)
            else:
                epistasis_pos.append(0)
            pairs_pos.append((pos1//3, pos2//3))

    return epistasis_nuc, pairs_nuc, epistasis_pos, pairs_pos

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
            epistasis_positions.append(np.max(epistasis_nucleotides[-9:]))
            position_pairs.append((pos1//3, pos2//3))
            

    return epistasis_nucleotides, nucleotide_pairs, epistasis_positions, position_pairs
