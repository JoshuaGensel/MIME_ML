import numpy as np

def preprocessSimData(path : str, maxSplits : int, readSize: int, seed = 1) -> np.array:
    
    """
    This function loads a sequence output of the dmMIMESim simulator, preprocesses it and returns it as a 2d numpy array. It parces the sequences to an 
    alignment of reads to emulate the output of the experimenal MIME procedure.
    
    Firstly it converts the sequences to a binary encoding where '1 = mutated', '-1 = wildtype'. Then it cuts the sequences into fragments at random 
    points, but not more then 'maxSplits'. The parts of the fragments that would not be read because of the limited read length ('readSize') get 
    replaced by zeros (Therefore '0 = missing'). The fragments get aligned onto the original sequence, where every residue that the fragments do not 
    cover are set to zero as well. Finally the resulting read alignment is returned as a 2d numpy array with dtype = int.
    
    Args:
        path (str): path to the .txt-file of the sequence output from dmMIMESim.
        maxSplits (int): maximum number of random splitting points.
        readSize (int): maximum read length of the sequencing machine (for Illumina deep sequencing = 100).
        seed (int, optional): numpy seed. important for reproducability of random fragmentation. Defaults to 1.

    Returns:
        np.array: _description_
    """
    
    
    
    #set seed
    np.random.seed(seed)
    
    #read in sequences
    with open(path, 'r') as f:
        sequenceList = []
        for line in f:
            sequenceList.append(line.strip())
            
    #define sequence length as constant
    seqLength = len(sequenceList[0])

    #fucntion that converts a sequence to a one-hot encoding
    def seq_to_bin(sequence):
        one_hot = []
        for i in range(len(sequence)):
            if sequence[i] == 'A':
                one_hot.append(-1)
            elif sequence[i] == 'C':
                one_hot.append(1)
        return one_hot
    
    #convert all sequences to one-hot encoding
    binarySequence = np.array([seq_to_bin(seq) for seq in sequenceList])
    
    #function that splits a sequence at up to 3 random points into fragments
    def splitSequence(seq: str) -> list:
        #split sequence at random points
        splitPoints = np.random.randint(1, len(seq), np.random.randint(1, maxSplits+1))
        splitPoints = np.sort(splitPoints)
        splitPoints = np.insert(splitPoints, 0, 0)
        splitPoints = np.append(splitPoints, len(seq))
        fragments = [seq[splitPoints[i] : splitPoints[i+1]] for i in range(len(splitPoints)-1)]
        
        return fragments

    #split all sequences into fragments
    fragmentList = [splitSequence(seq) for seq in binarySequence]
    
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
    alignedFragmentsMatrix = np.stack([item for sublist in alignedFragmentsList for item in sublist], axis = 0).astype(int)
    
        
    return alignedFragmentsMatrix
        