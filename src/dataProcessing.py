import numpy as np
from tqdm import tqdm
import os
import random


#function that splits a sequence at binomial random points into fragments
def splitSequence(seq: str, sequenceLength: int, splittingProbability: float) -> list:
    """
    splits a sequence at binomial random points into fragments

    Args:
        seq (str): sequence to be split.
        sequenceLength (int): length of the sequence.
        splittingProbability (float): probability of splitting per nucleotide.

    Returns:
        list: list of fragments.
    """
    #generate binomial random splits
    nSplits = np.random.binomial(sequenceLength, splittingProbability)
    splitPoints = np.random.randint(1, len(seq), nSplits)
    #sort split points
    splitPoints = np.sort(splitPoints)
    #add 0 and sequence length to split points
    splitPoints = np.insert(splitPoints, 0, 0)
    splitPoints = np.append(splitPoints, len(seq))
    #split sequence at split points
    fragments = [seq[splitPoints[i] : splitPoints[i+1]] for i in range(len(splitPoints)-1)]
    
    return fragments

#function that takes fragments and replaces values that are out of read range with 0
def replaceOutRange(fragments: list, readRange: int) -> list:
    """
    Takes fragments and replaces values that are out of read range with 0.
    A fragment is read in from both ends. If the fragment is longer then double the read range, the values in the middle are replaced with 0.

    Args:
        fragments (list): list of fragments. should be the output of splitSequence.
        readRange (int): maximum possible length of reads.

    Returns:
        list: list of fragments with values that are out of read range replaced with 0.
    """
    readFragments = fragments
    #if values are more than readRange away from start or end of fragment replace with 0
    for i in readFragments:
        #if whole fragment can be read in, do nothing
        if len(i) < readRange*2:
            pass
        #replace out of read range values with zeros
        i[readRange:-readRange] = 0
    return readFragments

#function that alignes fragments to sequence
def alignFragments(readFragments: list, sequenceLength: int) -> list:
    """
    Aligns fragments to sequence and puts 0 for missing values.

    Args:
        readFragments (list): list of read fragments(out of read range values should be missing). should be the output of replaceOutRange.
        sequenceLength (int): length of the sequence.

    Returns:
        list: list of aligned fragments.
    """
    #align sequences and put 0 for missing values
    #for all fragments add zeros in the beginning for length of all previous fragments
    alignedFragments = readFragments
    for i in range(1, len(readFragments)):
        nZeros = len(readFragments[i-1])
        alignedFragments[i] = np.insert(readFragments[i], 0, np.zeros(nZeros))
    # add zeros to the end of all fragments till length of sequence
    for i in range(len(readFragments)):
        nZeros = sequenceLength - len(readFragments[i])
        alignedFragments[i] = np.insert(readFragments[i], len(readFragments[i]), np.zeros(nZeros))
          
    
    return alignedFragments

def convertToMatrix(path : str, maxNumSequences : int, splittingProbability : float, readSize: int, seed = 1) -> np.array:
    
    """
    This function loads a sequence output of the dmMIMESim simulator, preprocesses it and returns it as a 2d numpy array. It parces the sequences to an 
    alignment of reads to emulate the output of the experimenal MIME procedure.
    
    Firstly it converts the sequences to a binary encoding where '1 = mutated', '-1 = wildtype'. Then it cuts the sequences into fragments at random 
    points, but not more then 'maxSplits'. The parts of the fragments that would not be read because of the limited read length ('readSize') get 
    replaced by zeros (Therefore '0 = missing'). The fragments get aligned onto the original sequence, where every residue that the fragments do not 
    cover are set to zero as well. Finally the resulting read alignment is returned as a 2d numpy array with dtype = int.
    
    Args:
        path (str): path to the .txt-file of the sequence output from dmMIMESim.
        maxNumSequences (int): The maximum number of sequences to be loaded. If the number of sequences in the file is smaller, all sequences are loaded.
        splittingProbability (float): The probability per nucleotide to pslit the sequence there into fragments.
        readSize (int): maximum read length of the sequencing machine (for Illumina deep sequencing = 100).
        proteinConcentration (int): protein concentration in the experiment.
        proteinConcentration2 (int): protein concentration in the second round experiment. If None/default it is not added to the array.
        seed (int, optional): numpy seed. important for reproducability of random fragmentation. Defaults to 1.

    Returns:
        np.array: 2d numpy array of fragmented sequence alignment.
    """
    
    
    
    #set seed
    np.random.seed(seed)
     
    #read in random subset of sequences
    sequenceArray = np.genfromtxt(path, dtype = str, max_rows = maxNumSequences, delimiter=1)
            
    #define sequence length as constant
    sequenceLength = len(sequenceArray[0])

    #split all sequences into fragments
    fragmentList = [splitSequence(seq, sequenceLength, splittingProbability) for seq in sequenceArray]
    
    #replace all values that are out of read range with 0 in all fragments
    readFragmentsList = [replaceOutRange(fragments, readSize) for fragments in fragmentList]
    
    #align all fragments to sequence
    alignedFragmentsList = [alignFragments(fragments, sequenceLength) for fragments in readFragmentsList]
    
    #create matrix of all aligned fragments
    alignedFragmentsMatrix = np.stack([item for sublist in alignedFragmentsList for item in sublist], axis = 0)

    return alignedFragmentsMatrix

def fragmentSequences(pathInput : str, pathOutput : str, maxNumSequences : int, splittingProbability : float, readSize: int, seed = 1) -> np.array:
    
    """
    This function loads a sequence output of the dmMIMESim simulator, preprocesses it and writes an output file from it. It parces the sequences to an 
    alignment of reads to emulate the output of the experimenal MIME procedure.
    
    Firstly it converts the sequences to a binary encoding where '1 = mutated', '-1 = wildtype'. Then it cuts the sequences into fragments at random 
    points, but not more then 'maxSplits'. The parts of the fragments that would not be read because of the limited read length ('readSize') get 
    replaced by zeros (Therefore '0 = missing'). The fragments get aligned onto the original sequence, where every residue that the fragments do not 
    cover are set to zero as well. Finally the resulting read alignment is written to the outputPath.
    
    Args:
        pathInput (str): path to the .txt-file of the sequence output from dmMIMESim.
        pathOutput (str): path to the output file.
        maxNumSequences (int): The maximum number of sequences to be loaded. If the number of sequences in the file is smaller, all sequences are loaded.
        splittingProbability (float): The probability per nucleotide to pslit the sequence there into fragments.
        readSize (int): maximum read length of the sequencing machine (for Illumina deep sequencing = 100).
        seed (int, optional): numpy seed. important for reproducability of random fragmentation. Defaults to 1.

    Returns:
        np.array: 2d numpy array of fragmented sequence alignment.
    """
    fragmentsMatrix = convertToMatrix(pathInput,maxNumSequences, splittingProbability, readSize, seed)
    np.savetxt(pathOutput, fragmentsMatrix, fmt='%s', delimiter='')
    return None

#function to mark wildtype residues
def markWildtype(sequences : np.array, pathToWildtypeSequence : str) -> np.array:
    """
    This function reads in a wildtype and replaces all wildtype residues of a provided sequence array with '1'. Then it returns the marked sequences as a numpy array.

    Args:
        sequences (np.array): 2d numpy array of sequences. dtype = str.
        pathToWildtypeSequence (str): path to the .txt-file of the wildtype sequence.

    Returns:
        np.array: 2d numpy array of marked sequences.
    """

    #read in wildtype sequence
    wildtypeSequence = np.genfromtxt(pathToWildtypeSequence, dtype = str, delimiter=1)

    #replace wildtype residues with 1
    markedSequenceArray = np.where(sequences == wildtypeSequence, 1, sequences)

    return markedSequenceArray


#function that reads in file and creates labels
def createLabels(boundPool : np.array, unboundPool : np.array) -> np.array:
    """
    This function takes in two sequence arrays for bound and unbound pool. It then creates the labels for all fragments and outputs a numpy array
    of the sequences and the labels. It takes both input arrays and concatenates them to one array. Then for every sequence it counts how often sequences 
    with identical mutations appear in the bound and unbound pool. The sequence is then labeled with the proportion of sequences with identical mutations 
    in the bound pool. If the sequence is not in the bound pool, it is labeled with 0. The resulting array is then returned.

    Args:
        boundPool (np.array): 2d numpy array of sequences of the bound pool. dtype = str.
        unboundPool (np.array): 2d numpy array of sequences of the unbound pool. dtype = str.

    Returns:
        np.array: 2d numpy array of sequences and labels.
    """

    #concatenate bound and unbound pool
    pool = np.concatenate((boundPool, unboundPool), axis = 0)

    #replace '1' with '0' in the pool
    poolOnlyMutations = np.where(pool == '1', '0', pool)
    boundPoolOnlyMutations = np.where(boundPool == '1', '0', boundPool)
    unboundPoolOnlyMutations = np.where(unboundPool == '1', '0', unboundPool)

    print('creating labels...')
    #create dicts for bound and unbound pool
    boundDict = {}
    unboundDict = {}

    #iterate through all unique sequences in the bound pool
    for seq in tqdm(np.unique(boundPoolOnlyMutations, axis = 0)):
        #for every sequence count how often it appears in the bound pool
        boundDict[seq.tobytes()] = np.count_nonzero(np.all(boundPoolOnlyMutations == seq, axis = 1))
    
    #iterate through all unique sequences in the unbound pool
    for seq in tqdm(np.unique(unboundPoolOnlyMutations, axis = 0)):
        #for every sequence count how often it appears in the unbound pool
        unboundDict[seq.tobytes()] = np.count_nonzero(np.all(unboundPoolOnlyMutations == seq, axis = 1))

    #create empty array for labels
    labels = np.zeros((pool.shape[0], 1))

    #iterate through all sequences in the pool
    for i in tqdm(range(pool.shape[0])):
        countBound = boundDict.get(poolOnlyMutations[i].tobytes(), 0)
        countUnbound = unboundDict.get(poolOnlyMutations[i].tobytes(), 0)
        #calculate the proportion of sequences with identical mutations in the bound pool
        if countBound + countUnbound != 0:
            labels[i] = countBound / (countBound + countUnbound)
            #print(i, '/', pool.shape[0], 'label', labels[i], 'countBound', countBound, 'countUnbound', countUnbound)
        else:
            labels[i] = 0
            #print(i, '/', pool.shape[0], 'label', labels[i], 'countBound', countBound, 'countUnbound', countUnbound)

    #add labels to pool
    pool = np.column_stack((pool, labels))

    return pool

#function that unmarks the wildtype sequences
def unmarkWildtype(sequences : np.array, pathToWildtype : str) -> np.array:
    """
    This function reads in a sequence file and a wildtype sequence file. It then unmarks the wildtype sequences in the sequence file and
    returns the resulting array. Wildtype residues in the input sequences are marked with a "1" and are converted back to the corresponding 
    letters and then returned.

    Args:
        pathToSequences (str): path to the .txt-file of the sequences.
        pathToWildtype (str): path to the .txt-file of the wildtype sequence.

    Returns:
        np.array: 2d numpy array of sequences.
    """

    #read in wildtype sequence
    wildtype = np.genfromtxt(pathToWildtype, dtype = str, delimiter=1)

    #iterate through all columns in the sequences except for the last one
    for i in range(sequences.shape[1] - 1):
        #replace every "1" with the corresponding letter in the wildtype sequence
        sequences[:,i] = np.where(sequences[:,i] == "1", wildtype[i], sequences[:,i])

    return sequences


def labelData(pathToBound : str, pathToUnbound : str, pathToWildtype : str, outputPath : str) -> None:
    """_summary_

    Args:
        pathToBound (str): _description_
        pathToUnbound (str): _description_
        pathToWildtype (str): _description_
        outputPath (str): _description_

    Returns:
        _type_: _description_
    """
    #read in the bound and unbound sequences
    bound = np.genfromtxt(pathToBound, dtype = str, delimiter=" ")
    unbound = np.genfromtxt(pathToUnbound, dtype = str, delimiter=" ")

    #mark wildtype in the sequences
    boundMarked = markWildtype(bound, pathToWildtype)
    unboundMarked = markWildtype(unbound, pathToWildtype)

    #label the sequences
    labelledSequences = createLabels(boundMarked, unboundMarked)

    #unmark the wildtype sequences
    unmarkedSequences = unmarkWildtype(labelledSequences, pathToWildtype)

    #write the sequences to a file
    np.savetxt(outputPath, unmarkedSequences, fmt='%s', delimiter=' ')

    return None

#function that rewrites the sequences to a one hot encoded format
def oneHotEncode(pathToData : str, outputPath : str) -> None:
    """
    This function reads a sequence file line by line and encodes each sequence to a one hot encoded format. The resulting sequences are then
    written to a file.

    Args:
        pathToData (str): path to the .txt-file of the sequences.
        outputPath (str): path to the output file.
    """

    #read line by line
    with open(pathToData, 'r') as f:
        with open(outputPath, 'w') as f2:
            for line in f:
                #split the line into a list
                line = line.split()
                #iterate through all elements in the list except for the last one
                for i in range(len(line)):
                    #replace every "0" with "0 0 0 0"
                    if line[i] == "0":
                        line[i] = "0 0 0 0"
                    #replace every "A" with "1 0 0 0"
                    if line[i] == "A":
                        line[i] = "1 0 0 0"
                    #replace every "G" with "0 1 0 0"
                    elif line[i] == "C":
                        line[i] = "0 1 0 0"
                    #replace every "C" with "0 0 1 0"
                    elif line[i] == "G":
                        line[i] = "0 0 1 0"
                    #replace every "T" with "0 0 0 1"
                    elif line[i] == "T":
                        line[i] = "0 0 0 1"
                #write the line to the file
                f2.write(" ".join(line))
                f2.write("\n")
    return None

#function to automatically parse dmMIMESim data
def parseSimDataPool(pathToData : str, outputPath : str, trainingSize : float, splittingProbability : float, readSize : int, seed : int) -> None:
    """
    This function reads in the dmMIMESim data files of one pool and parses them to training data. Then the data is written to a file.

    Args:
        pathToData (str): path to the simulator output directory from simAutomation.py.
        outputPath (str): path to the output file.
        trainingSize (float): fraction of the data to be used for training.
        splittingProbability (float): probability of splitting at each position.
        readSize (int): read size of the sequencer that reads in fragments from both sides
        seed (int): seed for the random number generator.
    """

    #fragment the sequences
    fragmentSequences(pathInput=pathToData+'7.txt', pathOutput="./data/simData/fragmentedDataBound.txt", maxNumSequences=1000000, splittingProbability= splittingProbability, readSize=readSize,seed=seed+42)
    fragmentSequences(pathInput=pathToData+'8.txt', pathOutput="./data/simData/fragmentedDataUnbound.txt", maxNumSequences=1000000, splittingProbability= splittingProbability, readSize=readSize,seed=seed+37)

    #read in the bound and unbound sequences
    bound = np.genfromtxt("./data/simData/fragmentedDataBound.txt", dtype = str, delimiter=1)
    unbound = np.genfromtxt("./data/simData/fragmentedDataUnbound.txt", dtype = str, delimiter=1)

    #shuffle the sequences
    np.random.seed(seed)
    np.random.shuffle(bound)
    np.random.seed(seed+67)
    np.random.shuffle(unbound)

    #split the sequences into training and test data
    boundTraining = bound[:int(trainingSize*bound.shape[0])]
    boundTest = bound[int(trainingSize*bound.shape[0]):]
    unboundTraining = unbound[:int(trainingSize*unbound.shape[0])]
    unboundTest = unbound[int(trainingSize*unbound.shape[0]):]

    #write the training and test data to a file
    np.savetxt('./data/simData/boundTraining.txt', boundTraining, fmt='%s')
    np.savetxt('./data/simData/boundTest.txt', boundTest, fmt='%s')
    np.savetxt('./data/simData/unboundTraining.txt', unboundTraining, fmt='%s')
    np.savetxt('./data/simData/unboundTest.txt', unboundTest, fmt='%s')

    #remove arrays from memory
    del bound
    del unbound
    del boundTraining
    del boundTest
    del unboundTraining
    del unboundTest

    #label the sequences
    labelData(pathToBound="./data/simData/boundTraining.txt", pathToUnbound="./data/simData/unboundTraining.txt", pathToWildtype="./data/simData/wildtype.txt", outputPath="./data/simData/labelledDataTraining.txt")
    labelData(pathToBound="./data/simData/boundTest.txt", pathToUnbound="./data/simData/unboundTest.txt", pathToWildtype="./data/simData/wildtype.txt", outputPath="./data/simData/labelledDataTest.txt")

    #one hot encode the sequences
    oneHotEncode(pathToData="./data/simData/labelledDataTraining.txt", outputPath=outputPath+'Training.txt')
    oneHotEncode(pathToData="./data/simData/labelledDataTest.txt", outputPath=outputPath+'Test.txt')

    print("Pool done!")

    return None


#parseSimDataPool(pathToData="/mnt/d/data/MIME_data/simData/dmMIME/seqLen100/experimentalConditions/secondFromProt1/prot6/sequences/", outputPath="./data/simData/", trainingSize=0.7, splittingProbability=0/100, readSize=50,seed=133)

#function to load all pools from a dmMIMESim run
def loadAllPools(pathToData : str, outputPath : str, proteinConcentrations : list, trainingSize : float, splittingProbability : float, readSize : int, seed : int) -> None:
    """
    This function loads all pools from a dmMIMESim run and parses them to training data. Then the data is written to a file.

    Args:
        pathToData (str): path to the simulator output directory from simAutomation.py.
        outputPath (str): path to the output file.
        trainingSize (float): fraction of the data to be used for training.
        splittingProbability (float): probability of splitting at each position.
        readSize (int): read size of the sequencer that reads in fragments from both sides
        seed (int): seed for the random number generator.
    """

    for firstRoundConcentration in proteinConcentrations:
        for secondRoundConcentration in proteinConcentrations:
            print("Loading pool with first round concentration " + str(firstRoundConcentration) + " and second round concentration " + str(secondRoundConcentration))
            pathToPool = pathToData + "secondFromProt" + str(firstRoundConcentration) + "/prot" + str(secondRoundConcentration) + "/sequences/"
            outputPathPool = outputPath + str(firstRoundConcentration) + "_" + str(secondRoundConcentration) + "/"
            #create output directory
            if not os.path.exists(outputPathPool):
                os.makedirs(outputPathPool)
            parseSimDataPool(pathToData=pathToPool, outputPath=outputPathPool, trainingSize=trainingSize, splittingProbability=splittingProbability, readSize=readSize,seed=seed)
    
    #create new .txt file with all training data
    with open(outputPath + "trainingData.txt", "w") as file:
        #concatenate all training data
        for firstRoundConcentration in proteinConcentrations:
            for secondRoundConcentration in proteinConcentrations:
                #create one hot protein identifier
                proteinIdentifier = "0" * len(proteinConcentrations)*2
                proteinIdentifier = proteinIdentifier[:proteinConcentrations.index(firstRoundConcentration)] + "1" + proteinIdentifier[proteinConcentrations.index(firstRoundConcentration)+1:]
                proteinIdentifier = proteinIdentifier[:len(proteinConcentrations) + proteinConcentrations.index(secondRoundConcentration)] + "1" + proteinIdentifier[len(proteinConcentrations) + proteinConcentrations.index(secondRoundConcentration)+1:]
                #put whitespace between each character
                proteinIdentifier = " ".join(proteinIdentifier)
                with open(outputPath + str(firstRoundConcentration) + "_" + str(secondRoundConcentration) + "/Training.txt", "r") as file2:
                    #add protein identifier to start of each line
                    file.write("".join([proteinIdentifier + " " + line for line in file2]))
        

    #create new .txt file with all test data
    with open(outputPath + "testData.txt", "w") as file:
        #concatenate all test data
        for firstRoundConcentration in proteinConcentrations:
            for secondRoundConcentration in proteinConcentrations:
                #create one hot protein identifier
                proteinIdentifier = "0" * len(proteinConcentrations)*2
                proteinIdentifier = proteinIdentifier[:proteinConcentrations.index(firstRoundConcentration)] + "1" + proteinIdentifier[proteinConcentrations.index(firstRoundConcentration)+1:]
                proteinIdentifier = proteinIdentifier[:len(proteinConcentrations) + proteinConcentrations.index(secondRoundConcentration)] + "1" + proteinIdentifier[len(proteinConcentrations) + proteinConcentrations.index(secondRoundConcentration)+1:]
                #put whitespace between each character
                proteinIdentifier = " ".join(proteinIdentifier)
                with open(outputPath + str(firstRoundConcentration) + "_" + str(secondRoundConcentration) + "/Test.txt", "r") as file2:
                    #add protein identifier to start of each line and write to file at random positions
                    file.write("".join([proteinIdentifier + " " + line for line in file2]))
    
    #shuffle the training data
    with open(outputPath + "trainingData.txt", "r") as file:
        lines = file.readlines()
        random.seed(seed)
        random.shuffle(lines)

    #write shuffled training data to file
    with open(outputPath + "trainingData.txt", "w") as file:
        file.write("".join(lines))
    
    #shuffle the test data
    with open(outputPath + "testData.txt", "r") as file:
        lines = file.readlines()
        random.seed(seed)
        random.shuffle(lines)

    #write shuffled test data to file
    with open(outputPath + "testData.txt", "w") as file:
        file.write("".join(lines))

    return None

loadAllPools(pathToData="/mnt/d/data/MIME_data/simData/dmMIME/seqLen100/experimentalConditions/", outputPath="/mnt/d/data/MIME_data/simData/dmMIME/seqLen100/experimentalConditions/data/", proteinConcentrations=[1,6,15,30], trainingSize=0.8, splittingProbability=3/100, readSize=23,seed=133)


# fragmentSequences(pathInput="./src/expFiles/bound.txt", pathOutput="./src/expFiles/fragmentedDataBound.txt", maxNumSequences=10000, splittingProbability= 1/20, readSize=7,seed=42)
# fragmentSequences(pathInput="./src/expFiles/unbound.txt", pathOutput="./src/expFiles/fragmentedDataUnbound.txt", maxNumSequences=10000, splittingProbability= 1/20, readSize=7,seed=42)
# labelData(pathToBound="./src/expFiles/fragmentedDataBound.txt", pathToUnbound="./src/expFiles/fragmentedDataUnbound.txt", pathToWildtype="./src/expFiles/wildtype.txt", outputPath="./src/expFiles/labelledData.txt")
# oneHotEncode(pathToData="./src/expFiles/labelledData.txt", outputPath="./src/expFiles/oneHotEncodedData.txt")