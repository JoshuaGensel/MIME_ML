import numpy as np

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


boundFragmented = convertToMatrix('./src/expFiles/bound.txt', maxNumSequences=100, splittingProbability=1/20, readSize=5, seed=1)
np.savetxt('./src/expFiles/boundFragmented.txt', boundFragmented, fmt='%s', delimiter='')
unboundFragmented = convertToMatrix('./src/expFiles/unbound.txt', maxNumSequences=100, splittingProbability=1/20, readSize=5, seed=372)
np.savetxt('./src/expFiles/unboundFragmented.txt', unboundFragmented, fmt='%s', delimiter='')

#function to mark wildtype residues
def markWildtype(pathToSequenceFile : str, pathToWildtypeSequence : str) -> np.array:
    """
    This function reads in sequence files and replaces all wildtype residues with '1'. Then it returns the marked sequences as a numpy array.

    Args:
        pathToSequenceFile (str): path to the .txt-file of the sequences.
        pathToWildtypeSequence (str): path to the .txt-file of the wildtype sequence.

    Returns:
        np.array: 2d numpy array of marked sequences.
    """

    #read in sequences and wildtype sequence
    sequenceArray = np.genfromtxt(pathToSequenceFile, dtype = str, delimiter=1)
    wildtypeSequence = np.genfromtxt(pathToWildtypeSequence, dtype = str, delimiter=1)

    #replace wildtype residues with 1
    markedSequenceArray = np.where(sequenceArray == wildtypeSequence, 1, sequenceArray)

    return markedSequenceArray

boundMarked = markWildtype('./src/expFiles/boundFragmented.txt', './src/expFiles/wildtype.txt')
np.savetxt('./src/expFiles/boundMarked.txt', boundMarked, fmt='%s', delimiter='')
unboundMarked = markWildtype('./src/expFiles/unboundFragmented.txt', './src/expFiles/wildtype.txt')
np.savetxt('./src/expFiles/unboundMarked.txt', unboundMarked, fmt='%s', delimiter='')


#function that reads in file and creates labels
def createLabels(pathBound : str, pathUnbound : str) -> np.array:
    """
    This function reads in two sequence fragment files from bound and unbound pool. It then creates the labels for all fragments and outputs a numpy array
    of the sequences and the labels. It reads in both files and concatenates them to one array. Then for every sequence it counts how often it appears in
    the bound and unbound pool. The sequence is then labeled with the proportion of this unique sequence in the bound pool. If the sequence is not in the
    bound pool, it is labeled with 0. The resulting array is then returned.

    Args:
        pathBound (str): path to the .txt-file of the bound pool.
        pathUnbound (str): path to the .txt-file of the unbound pool.

    Returns:
        np.array: 2d numpy array of sequences and labels.
    """

    #read in bound and unbound pool
    boundPool = np.genfromtxt(pathBound, dtype = str, delimiter=1)
    unboundPool = np.genfromtxt(pathUnbound, dtype = str, delimiter=1)

    #concatenate bound and unbound pool
    pool = np.concatenate((boundPool, unboundPool), axis = 0)

    countWildtypeBound = 0
    countWildtypeUnbound = 0

    #count the number of wildtype sequences in the bound pool
    for i in range(len(boundPool)):
        #if sequence does not contain any "A", "G", "C" or "T", count it
        if "A" not in boundPool[i] and "G" not in boundPool[i] and "C" not in boundPool[i] and "T" not in boundPool[i]:
            countWildtypeBound += 1
    #count the number of wildtype sequences in the unbound pool
    for i in range(len(unboundPool)):
        #if sequence does not contain any "A", "G", "C" or "T", count it
        if "A" not in unboundPool[i] and "G" not in unboundPool[i] and "C" not in unboundPool[i] and "T" not in unboundPool[i]:
            countWildtypeUnbound += 1
    wildtypeProportion = countWildtypeBound / (countWildtypeBound + countWildtypeUnbound)

    labels = np.array([])
    #iterate through all sequences in the pool
    for i in range(len(pool)):
        #if the sequence does not contain any "A", "G", "C" or "T", it is a wildtype sequence
        if np.array_equal(np.where(pool[i] == "A"), np.array([])) and np.array_equal(np.where(pool[i] == "G"), np.array([])) and np.array_equal(np.where(pool[i] == "C"), np.array([])) and np.array_equal(np.where(pool[i] == "T"), np.array([])):
            labels = np.append(labels, wildtypeProportion)

        else:
            countBound = 0
            #iterate through all sequences in the bound pool
            for j in range(len(boundPool)):
                    #if the position of "A"s in the sequence is the same, count it
                    if np.array_equal(np.where(pool[i] == "A"), np.where(boundPool[j] == "A")):
                        #if the position of "G"s in the sequence is the same, count it
                        if np.array_equal(np.where(pool[i] == "G"), np.where(boundPool[j] == "G")):
                            #if the position of "C"s in the sequence is the same, count it
                            if np.array_equal(np.where(pool[i] == "C"), np.where(boundPool[j] == "C")):
                                #if the position of "T"s in the sequence is the same, count it
                                if np.array_equal(np.where(pool[i] == "T"), np.where(boundPool[j] == "T")):
                                    countBound += 1
            
            countUnbound = 0
            #iterate through all sequences in the unbound pool
            for j in range(len(unboundPool)):
                    #if the position of "A"s in the sequence is the same, count it
                    if np.array_equal(np.where(pool[i] == "A"), np.where(unboundPool[j] == "A")):
                        #if the position of "G"s in the sequence is the same, count it
                        if np.array_equal(np.where(pool[i] == "G"), np.where(unboundPool[j] == "G")):
                            #if the position of "C"s in the sequence is the same, count it
                            if np.array_equal(np.where(pool[i] == "C"), np.where(unboundPool[j] == "C")):
                                #if the position of "T"s in the sequence is the same, count it
                                if np.array_equal(np.where(pool[i] == "T"), np.where(unboundPool[j] == "T")):
                                    countUnbound += 1

            #label the sequences with the proportion of the unique sequence in the bound pool
            if countBound == 0:
                labels = np.append(labels, 0)
            else:
                labels = np.append(labels, countBound / (countBound + countUnbound))

    #add the labels to the pool
    pool = np.column_stack((pool, labels))

    return pool

dataMarked = createLabels('./src/expFiles/boundMarked.txt', './src/expFiles/unboundMarked.txt')
np.savetxt('./src/expFiles/dataMarked.txt', dataMarked, fmt='%s')

#function that unmarks the wildtype sequences
def unmarkWildtype(pathToSequences : str, pathToWildtype : str) -> np.array:
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

    #read in sequences and wildtype sequence
    sequences = np.genfromtxt(pathToSequences, dtype = str, delimiter=" ")
    wildtype = np.genfromtxt(pathToWildtype, dtype = str, delimiter=1)

    #iterate through all columns in the sequences except for the last one
    for i in range(sequences.shape[1] - 1):
        #replace every "1" with the corresponding letter in the wildtype sequence
        sequences[:,i] = np.where(sequences[:,i] == "1", wildtype[i], sequences[:,i])

    return sequences

dataUnmarked = unmarkWildtype('./src/expFiles/dataMarked.txt', './src/expFiles/wildtype.txt')
np.savetxt('./src/expFiles/dataUnmarked.txt', dataUnmarked, fmt='%s')

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

oneHotEncode('./src/expFiles/dataUnmarked.txt', './src/expFiles/dataOneHot.txt')