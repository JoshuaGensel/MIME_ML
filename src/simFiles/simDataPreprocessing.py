import numpy as np
from sklearn.preprocessing import PolynomialFeatures

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
        alignedFragments[i] = np.append(readFragments[i], np.zeros(nZeros))    
    
    return alignedFragments


def convertToMatrix(path : str, maxNumSequences : int, splittingProbability : float, readSize: int, proteinConcentration: int, proteinConcentration2 = None, seed = 1) -> np.array:
    
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

    #convert to binary encoding
    sequenceArray = np.where(sequenceArray == 'C', 1, -1).astype(int)

    #split all sequences into fragments
    fragmentList = [splitSequence(seq, sequenceLength, splittingProbability) for seq in sequenceArray]
    
    #replace all values that are out of read range with 0 in all fragments
    readFragmentsList = [replaceOutRange(fragments, readSize) for fragments in fragmentList]
    
    #align all fragments to sequence
    alignedFragmentsList = [alignFragments(fragments, sequenceLength) for fragments in readFragmentsList]
    
    #create matrix of all aligned fragments
    alignedFragmentsMatrix = np.stack([item for sublist in alignedFragmentsList for item in sublist], axis = 0)
    
    #if proteinConcentration2 is not None add it to the array
    if proteinConcentration2 != None:
        alignedFragmentsMatrix = np.insert(alignedFragmentsMatrix, 0, proteinConcentration2, axis = 1)
    
    #add protein concentration to each row
    alignedFragmentsMatrix = np.insert(alignedFragmentsMatrix, 0, proteinConcentration, axis = 1).astype(int)
        
    return alignedFragmentsMatrix

def convertToInteractionMatrix(path : str, maxNumSequences : int, splittingProbability : float, readSize: int, proteinConcentration: int, proteinConcentration2 = None, seed = 1) -> np.array:
    """
    This function loads a sequence output of the dmMIMESim simulator, preprocesses it and returns it as a 2d numpy array. It parces the sequences to an 
    alignment of reads to emulate the output of the experimenal MIME procedure. It also adds all pairwise interaction terms between the residues to the 
    array.
    
    Firstly it converts the sequences to a binary encoding where '1 = mutated', '-1 = wildtype'. Then it cuts the sequences into fragments at random 
    points, but not more then 'maxSplits'. The parts of the fragments that would not be read because of the limited read length ('readSize') get 
    replaced by zeros (Therefore '0 = missing'). The fragments get aligned onto the original sequence, where every residue that the fragments do not 
    cover are set to zero as well. Then for all residues the pairwise interaction terms are added to the array. Finally the resulting read alignment is 
    returned as a 2d numpy array with dtype = int.
    
    Args:
        path (str): path to the .txt-file of the sequence output from dmMIMESim.
        maxNumSequences (int): The maximum number of sequences to be loaded. If the number of sequences in the file is smaller, all sequences are loaded.
        splittingProbability (float): The probability per nucleotide to pslit the sequence there into fragments.
        readSize (int): maximum read length of the sequencing machine (for Illumina deep sequencing = 100).
        proteinConcentration (int): protein concentration in the experiment.
        proteinConcentration2 (int): protein concentration in the second round experiment. If None/default it is not added to the array.
        seed (int, optional): numpy seed. important for reproducability of random fragmentation. Defaults to 1.

    Returns:
        np.array: 2d numpy array of fragmented sequence alignment with all pairwise interaction terms.
    """
    
    #set seed
    np.random.seed(seed)
     
    #read in random subset of sequences
    sequenceArray = np.genfromtxt(path, dtype = str, max_rows = maxNumSequences, delimiter=1)
            
    #define sequence length as constant
    sequenceLength = len(sequenceArray[0])

    #convert to binary encoding
    sequenceArray = np.where(sequenceArray == 'C', 1, -1).astype(int)

    #split all sequences into fragments
    fragmentList = [splitSequence(seq, sequenceLength, splittingProbability) for seq in sequenceArray]
    
    #replace all values that are out of read range with 0 in all fragments
    readFragmentsList = [replaceOutRange(fragments, readSize) for fragments in fragmentList]
    
    #align all fragments to sequence
    alignedFragmentsList = [alignFragments(fragments, sequenceLength) for fragments in readFragmentsList]
    
    #create matrix of all aligned fragments
    alignedFragmentsMatrix = np.stack([item for sublist in alignedFragmentsList for item in sublist], axis = 0)
    
    #add interaction terms for every residue to the matrix
    interaction = PolynomialFeatures(interaction_only=True, include_bias=False)
    interactionMatrix = interaction.fit_transform(alignedFragmentsMatrix)
    
    #if proteinConcentration2 is not None add it to the rows
    if proteinConcentration2 != None:
        interactionMatrix = np.insert(interactionMatrix, 0, proteinConcentration2, axis = 1)
    
    #add protein concentration to each row
    interactionMatrix = np.insert(interactionMatrix, 0, proteinConcentration, axis = 1).astype(int)
    
    return interactionMatrix
    
def convertToOneHotMatrix(path : str, maxNumSequences : int, splittingProbability : float, readSize: int, proteinConcentration: int, proteinConcentration2 = None, seed = 1) -> np.array:
    """
    This function loads a sequence output of the dmMIMESim simulator, preprocesses it and returns it as a 2d numpy array. It parces the sequences to an 
    alignment of reads to emulate the output of the experimenal MIME procedure. It encodes all residues with a one hot encoding.
    
    Firstly it converts the sequences to a binary encoding where '1 = mutated', '-1 = wildtype'. Then it cuts the sequences into fragments at random 
    points, but not more then 'maxSplits'. The parts of the fragments that would not be read because of the limited read length ('readSize') get 
    replaced by zeros (Therefore '0 = missing'). The fragments get aligned onto the original sequence, where every residue that the fragments do not 
    cover are set to zero as well. Then for all residues the pairwise interaction terms are added to the array. Finally the resulting read alignment is 
    returned as a 2d numpy array with dtype = int.
    
    Args:
        path (str): path to the .txt-file of the sequence output from dmMIMESim.
        maxNumSequences (int): The maximum number of sequences to be loaded. If the number of sequences in the file is smaller, all sequences are loaded.
        splittingProbability (float): The probability per nucleotide to pslit the sequence there into fragments.
        readSize (int): maximum read length of the sequencing machine (for Illumina deep sequencing = 100).
        proteinConcentration (int): protein concentration in the experiment.
        proteinConcentration2 (int): protein concentration in the second round experiment. If None/default it is not added to the array.
        seed (int, optional): numpy seed. important for reproducability of random fragmentation. Defaults to 1.

    Returns:
        np.array: 2d numpy array of fragmented sequence alignment with all pairwise interaction terms.
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
    
    #create matrix of all aligned fragments
    alignedFragmentsMatrix = np.stack([item for sublist in alignedFragmentsList for item in sublist], axis = 0)
    
    #transform to one hot encoding
    oneHotMatrix = np.where(alignedFragmentsMatrix == '0.0' , '0,0,0,0', alignedFragmentsMatrix)
    oneHotMatrix = np.where(oneHotMatrix == '0' , '0,0,0,0', oneHotMatrix)
    oneHotMatrix = np.where(oneHotMatrix == 'T' , '0,0,0,1', oneHotMatrix)
    oneHotMatrix = np.where(oneHotMatrix == 'G' , '0,0,1,0', oneHotMatrix)
    oneHotMatrix = np.where(oneHotMatrix == 'C' , '0,1,0,0', oneHotMatrix)
    oneHotMatrix = np.where(oneHotMatrix == 'A' , '1,0,0,0', oneHotMatrix)
    oneHotMatrix = np.char.split(oneHotMatrix, sep = ',')
    
    #make array of lists into 3d array
    oneHotMatrix = np.stack([np.stack([np.array(item) for item in sublist], axis = 0) for sublist in oneHotMatrix], axis = 0)
    
    #convert to int
    oneHotMatrix = oneHotMatrix.astype(int)
    
    #flatten 3d array to 2d array
    oneHotMatrix = oneHotMatrix.reshape(oneHotMatrix.shape[0], oneHotMatrix.shape[1] * oneHotMatrix.shape[2])
    
    #if proteinConcentration2 is not None add it to the rows
    if proteinConcentration2 != None:
        oneHotMatrix = np.insert(oneHotMatrix, 0, proteinConcentration2, axis = 1)
    
    #add protein concentration to each row
    oneHotMatrix = np.insert(oneHotMatrix, 0, proteinConcentration, axis = 1).astype(int)
    
    
    return oneHotMatrix

def loadSimDataOneHot(path : str, maxNumSequences : int, splittingProbability : float, readSize: int):
    """
    This function loads the simulated data from the dmMIMESim simulator and returns training data and label data as numpy arrays. 
    It loads the data from all rounds from the directory used as output directory by the simAutomation.py script. The data gets
    parsed with convertToOneHotMatrix() and then split into training and label data. The training data is the one hot encoded
    array of aligned fragmented sequences. The label data is 1 if the sequence is bound and 0 if it is not bound. The data is
    not yet split or shuffled.

    Args:
        path (str): path to the directory where the simulated data is stored. Same as the output directory used by the simAutomation.py script.
        maxNumSequences (int): maximum number of sequences to be loaded from each round.
        splittingProbability (float): The probability per nucleotide to pslit the sequence there into fragments.
        readSize (int): maximum read length of the sequencing machine (for Illumina deep sequencing = 100).
        seed (int, optional): numpy seed. important for reproducability of random fragmentation. Defaults to 1.

    Returns:
        np.array, np.array: training data, label data
    """
    
    bound_1_1 = convertToOneHotMatrix(path+'/secondFromProt1/prot1/sequences/7.txt',
                            maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=1, proteinConcentration2=1, seed=2536)
    bound_1_6 = convertToOneHotMatrix(path+'/secondFromProt1/prot6/sequences/7.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=1, proteinConcentration2=6, seed=5675)
    bound_1_15 = convertToOneHotMatrix(path+'/secondFromProt1/prot15/sequences/7.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=1, proteinConcentration2=15, seed=3456)
    bound_1_30 = convertToOneHotMatrix(path+'/secondFromProt1/prot30/sequences/7.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=1, proteinConcentration2=30, seed=972358)

    unbound_1_1 = convertToOneHotMatrix(path+'/secondFromProt1/prot1/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=1, proteinConcentration2=1, seed=867532)
    unbound_1_6 = convertToOneHotMatrix(path+'/secondFromProt1/prot6/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=1, proteinConcentration2=6, seed=10357)
    unbound_1_15 = convertToOneHotMatrix(path+'/secondFromProt1/prot15/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=1, proteinConcentration2=15, seed=67238)
    unbound_1_30 = convertToOneHotMatrix(path+'/secondFromProt1/prot30/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=1, proteinConcentration2=30, seed=564123)
    print('first round prot1 done')
    bound_6_1 = convertToOneHotMatrix(path+'/secondFromProt6/prot1/sequences/7.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=6, proteinConcentration2=1, seed=4527)
    bound_6_6 = convertToOneHotMatrix(path+'/secondFromProt6/prot6/sequences/7.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=6, proteinConcentration2=6, seed=24577)
    bound_6_15 = convertToOneHotMatrix(path+'/secondFromProt6/prot15/sequences/7.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=6, proteinConcentration2=15, seed=23456)
    bound_6_30 = convertToOneHotMatrix(path+'/secondFromProt6/prot30/sequences/7.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=6, proteinConcentration2=30, seed=75948)

    unbound_6_1 = convertToOneHotMatrix(path+'/secondFromProt6/prot1/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=6, proteinConcentration2=1, seed=2456)
    unbound_6_6 = convertToOneHotMatrix(path+'/secondFromProt6/prot6/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=6, proteinConcentration2=6, seed=45678)
    unbound_6_15 = convertToOneHotMatrix(path+'/secondFromProt6/prot15/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=6, proteinConcentration2=15, seed=3415)
    unbound_6_30 = convertToOneHotMatrix(path+'/secondFromProt6/prot30/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=6, proteinConcentration2=30, seed=7469)
    print('first round prot6 done')
    bound_15_1 = convertToOneHotMatrix(path+'/secondFromProt15/prot1/sequences/7.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=15, proteinConcentration2=1, seed=13465)
    bound_15_6 = convertToOneHotMatrix(path+'/secondFromProt15/prot6/sequences/7.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=15, proteinConcentration2=6, seed=3658)
    bound_15_15 = convertToOneHotMatrix(path+'/secondFromProt15/prot15/sequences/7.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=15, proteinConcentration2=15, seed=13456)
    bound_15_30 = convertToOneHotMatrix(path+'/secondFromProt15/prot30/sequences/7.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=15, proteinConcentration2=30, seed=2457)

    unbound_15_1 = convertToOneHotMatrix(path+'/secondFromProt15/prot1/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=15, proteinConcentration2=1, seed=83656)
    unbound_15_6 = convertToOneHotMatrix(path+'/secondFromProt15/prot6/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=15, proteinConcentration2=6, seed=2574)
    unbound_15_15 = convertToOneHotMatrix(path+'/secondFromProt15/prot15/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=15, proteinConcentration2=15, seed=4679)
    unbound_15_30 = convertToOneHotMatrix(path+'/secondFromProt15/prot30/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=15, proteinConcentration2=30, seed=1345)
    print('first round prot15 done')
    bound_30_1 = convertToOneHotMatrix(path+'/secondFromProt30/prot1/sequences/7.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=30, proteinConcentration2=1, seed=3567)
    bound_30_6 = convertToOneHotMatrix(path+'/secondFromProt30/prot6/sequences/7.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=30, proteinConcentration2=6, seed=565674275)
    bound_30_15 = convertToOneHotMatrix(path+'/secondFromProt30/prot15/sequences/7.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=30, proteinConcentration2=15, seed=2457)
    bound_30_30 = convertToOneHotMatrix(path+'/secondFromProt30/prot30/sequences/7.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=30, proteinConcentration2=30, seed=25478)

    unbound_30_1 = convertToOneHotMatrix(path+'/secondFromProt30/prot1/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=30, proteinConcentration2=1, seed=6583)
    unbound_30_6 = convertToOneHotMatrix(path+'/secondFromProt30/prot6/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=30, proteinConcentration2=6, seed=4136)
    unbound_30_15 = convertToOneHotMatrix(path+'/secondFromProt30/prot15/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=30, proteinConcentration2=15, seed=4356)
    unbound_30_30 = convertToOneHotMatrix(path+'/secondFromProt30/prot30/sequences/8.txt',
                                maxNumSequences=maxNumSequences, splittingProbability=splittingProbability, readSize=readSize, proteinConcentration=30, proteinConcentration2=30, seed=564123)
    print('first round prot 30 done')
    
    #combine all bound and unbound data into one array
    bound = np.concatenate((bound_1_1, bound_1_6, bound_1_15, bound_1_30, bound_6_1, bound_6_6, bound_6_15, bound_6_30, bound_15_1, bound_15_6, bound_15_15, bound_15_30, bound_30_1, bound_30_6, bound_30_15, bound_30_30), axis=0)
    unbound = np.concatenate((unbound_1_1, unbound_1_6, unbound_1_15, unbound_1_30, unbound_6_1, unbound_6_6, unbound_6_15, unbound_6_30, unbound_15_1, unbound_15_6, unbound_15_15, unbound_15_30, unbound_30_1, unbound_30_6, unbound_30_15, unbound_30_30), axis=0)

    #create labels for bound and unbound data
    bound_labels = np.ones((bound.shape[0],1))
    unbound_labels = np.zeros((unbound.shape[0],1))

    #combine bound and unbound data and labels into one array
    data = np.concatenate((bound, unbound), axis=0)
    labels = np.concatenate((bound_labels, unbound_labels), axis=0)
    
    return data, labels
    