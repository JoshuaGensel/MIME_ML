import numpy as np
from numba import jit
from sklearn.preprocessing import PolynomialFeatures

@jit(parallel=True, forceobj=True)
def convertToMatrix(path : str, maxNumSequences : int, splitProb : float, readSize: int, protConc: int, protConc2 = None, seed = 1) -> np.array:
    
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
        protConc (int): protein concentration in the experiment.
        protConc2 (int): protein concentration in the second round experiment. If None/default it is not added to the array.
        seed (int, optional): numpy seed. important for reproducability of random fragmentation. Defaults to 1.

    Returns:
        np.array: 2d numpy array of fragmented sequence alignment.
    """
    
    
    
    #set seed
    np.random.seed(seed)
     
    #read in random subset of sequences
    sequenceArray = np.genfromtxt(path, dtype = str, max_rows = maxNumSequences, delimiter=1)
            
    #define sequence length as constant
    seqLength = len(sequenceArray[0])

    #convert to binary encoding
    sequenceArray = np.where(sequenceArray == 'C', 1, -1).astype(int)
    
    
    #function that splits a sequence at binomial random points into fragments
    def splitSequence(seq: str) -> list:
        #split sequence at random points
        nSplits = np.random.binomial(seqLength, splitProb)
        splitPoints = np.random.randint(1, len(seq), nSplits)
        splitPoints = np.sort(splitPoints)
        splitPoints = np.insert(splitPoints, 0, 0)
        splitPoints = np.append(splitPoints, len(seq))
        fragments = [seq[splitPoints[i] : splitPoints[i+1]] for i in range(len(splitPoints)-1)]
        
        return fragments

    #split all sequences into fragments
    fragmentList = [splitSequence(seq) for seq in sequenceArray]
    
    #function that takes fragments and replaces values that are out of read range with 0
    def replaceOutRange(fragments: list, readRange: int) -> list:
        readFragments = fragments
        #if values are more than readRange away from start or end of fragment replace with 0
        for i in readFragments:
            if len(i) < readRange*2:
                pass
            #replace out of read range values with zeros
            i[readRange:-readRange] = 0
        return readFragments
    
    #replace all values that are out of read range with 0 in all fragments
    readFragmentsList = [replaceOutRange(fragments, readSize) for fragments in fragmentList]
    
    #function that alignes fragments to sequence
    def alignFragments(readFragments: list, seqLength: int) -> list:
    #align sequences and put 0 for missing values
    #for all fragments add zeros in the beginning for length of all previous fragments
        alignedFragments = readFragments
        for i in range(1, len(readFragments)):
            numZeros = len(readFragments[i-1])
            alignedFragments[i] = np.insert(readFragments[i], 0, np.zeros(numZeros))
        # add zeros to the end of all fragments till length of sequence
        for i in range(len(readFragments)):
            numZeros = seqLength - len(readFragments[i])
            alignedFragments[i] = np.append(readFragments[i], np.zeros(numZeros))    
        
        return alignedFragments
    
    #align all fragments to sequence
    alignedFragmentsList = [alignFragments(fragments, seqLength) for fragments in readFragmentsList]
    
    #create matrix of all aligned fragments
    alignedFragmentsMatrix = np.stack([item for sublist in alignedFragmentsList for item in sublist], axis = 0)
    
    #if protConc2 is not None add it to the array
    if protConc2 != None:
        alignedFragmentsMatrix = np.insert(alignedFragmentsMatrix, 0, protConc2, axis = 1)
    
    #add protein concentration to each row
    alignedFragmentsMatrix = np.insert(alignedFragmentsMatrix, 0, protConc, axis = 1).astype(int)
        
    return alignedFragmentsMatrix


def convertToInteractionMatrix(path : str, maxNumSequences : int, splitProb : float, readSize: int, protConc: int, protConc2 = None, seed = 1) -> np.array:
    #set seed
    np.random.seed(seed)
     
    #read in random subset of sequences
    sequenceArray = np.genfromtxt(path, dtype = str, max_rows = maxNumSequences, delimiter=1)
            
    #define sequence length as constant
    seqLength = len(sequenceArray[0])

    #convert to binary encoding
    sequenceArray = np.where(sequenceArray == 'C', 1, -1).astype(int)
    
    
    #function that splits a sequence at binomial random points into fragments
    def splitSequence(seq: str) -> list:
        #split sequence at random points
        nSplits = np.random.binomial(seqLength, splitProb)
        splitPoints = np.random.randint(1, len(seq), nSplits)
        splitPoints = np.sort(splitPoints)
        splitPoints = np.insert(splitPoints, 0, 0)
        splitPoints = np.append(splitPoints, len(seq))
        fragments = [seq[splitPoints[i] : splitPoints[i+1]] for i in range(len(splitPoints)-1)]
        
        return fragments

    #split all sequences into fragments
    fragmentList = [splitSequence(seq) for seq in sequenceArray]
    
    #function that takes fragments and replaces values that are out of read range with 0
    def replaceOutRange(fragments: list, readRange: int) -> list:
        readFragments = fragments
        #if values are more than readRange away from start or end of fragment replace with 0
        for i in readFragments:
            if len(i) < readRange*2:
                pass
            #replace out of read range values with zeros
            i[readRange:-readRange] = 0
        return readFragments
    
    #replace all values that are out of read range with 0 in all fragments
    readFragmentsList = [replaceOutRange(fragments, readSize) for fragments in fragmentList]
    
    #function that alignes fragments to sequence
    def alignFragments(readFragments: list, seqLength: int) -> list:
    #align sequences and put 0 for missing values
    #for all fragments add zeros in the beginning for length of all previous fragments
        alignedFragments = readFragments
        for i in range(1, len(readFragments)):
            numZeros = len(readFragments[i-1])
            alignedFragments[i] = np.insert(readFragments[i], 0, np.zeros(numZeros))
        # add zeros to the end of all fragments till length of sequence
        for i in range(len(readFragments)):
            numZeros = seqLength - len(readFragments[i])
            alignedFragments[i] = np.append(readFragments[i], np.zeros(numZeros))    
        
        return alignedFragments
    
    #align all fragments to sequence
    alignedFragmentsList = [alignFragments(fragments, seqLength) for fragments in readFragmentsList]
    
    #create matrix of all aligned fragments
    alignedFragmentsMatrix = np.stack([item for sublist in alignedFragmentsList for item in sublist], axis = 0)
    
    #add interaction terms for every residue to the matrix
    interaction = PolynomialFeatures(interaction_only=True, include_bias=False)
    interactionMatrix = interaction.fit_transform(alignedFragmentsMatrix)
    
    #if protConc2 is not None add it to the rows
    if protConc2 != None:
        interactionMatrix = np.insert(interactionMatrix, 0, protConc2, axis = 1)
    
    #add protein concentration to each row
    interactionMatrix = np.insert(interactionMatrix, 0, protConc, axis = 1).astype(int)
    
    return interactionMatrix
    