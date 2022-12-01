import numpy as np
from numba import jit
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

@jit(parallel=True, forceobj=True)
def convertToMatrix(path : str, maxNumSequences : int, splittingProbability : float, readSize: int, proteinConcentration: int, proteinConcentration2 = None, seed = 1) -> np.array:
    
    """
    This function loads a sequence output of the dmMIMESim simulator, preprocesses it and returns it as a 2d numpy array. It parces the sequences to an 
    alignment of reads to emulate the output of the experimenal MIME procedure.
    
    Firstly it converts the sequences to a binary encoding where '1 = mutated', '-1 = wildtype'. Then it cuts the sequences into fragments at random 
    points, but not more then 'maxSplits'. The parts of the fragments that would not be read because of the limited read length ('readSize') get 
    replaced by zeros (Therefore '0 = missing'). The fragments get aligned onto the original sequence, where every residue that the fragments do not 
    cover are set to zero as well. Finally the resulting read alignment is returned as a 2d numpy array with dtype = int.
    
    Args:
        maxNumSequences (int): The maximum number of sequences to be loaded. If the number of sequences in the file is smaller, all sequences are loaded.
        path (str): path to the .txt-file of the sequence output from dmMIMESim.
        maxSplits (int): maximum number of random splitting points.
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

@jit(parallel=True, forceobj=True)
def convertToInteractionMatrix(path : str, maxNumSequences : int, splittingProbability : float, readSize: int, proteinConcentration: int, proteinConcentration2 = None, seed = 1) -> np.array:
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
    
@jit(parallel=True, forceobj=True)
def convertToOneHotMatrix(path : str, maxNumSequences : int, splittingProbability : float, readSize: int, proteinConcentration: int, proteinConcentration2 = None, seed = 1) -> np.array:
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
    oneHotMatrix = np.where(oneHotMatrix == 'U' , '0,0,0,1', oneHotMatrix)
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
