import re
import shutil
import numpy as np
from tqdm import tqdm
import os 
import random

def align_reads_simulator(file_path_reads: str, file_path_output: str, splitting_probabality: float, read_length: int, seed: int):
    """
    This function takes the full sequences that are output by the MIME Simulator
    and creates aligned reads of the fragments of this sequence. First the full 
    sequences from the simulator are split into fragments. The probability of 
    a split happening at a given position is given by the splitting_probability.
    The fragments are then written in from both sides until the read_length is
    reached. Missing positions that are not covered by the fragments or that are
    out of range are filled with zeros. The aligned reads are written to the 
    output file (file_path_output).

    Args:
        file_path_reads (str): file path to the full sequences output by the 
            MIME Simulator (sequences/7.txt or sequences/8.txt)
        file_path_output (str): file path to the output file 
            (aligned_reads_bound.txt or aligned_reads_unbound.txt)
        splitting_probability (float): probability of a split happening at a
            given position
        read_length (int): maximum length that a fragment can be read in (from
            each side)
        seed (int): seed for the random number generator
    """

    # set the seed
    random.seed(seed)

    # read in the full sequences line by line
    with open(file_path_reads, 'r') as reads_file:
        reads = reads_file.readlines()
        # remove the new line character at the end of each line
        reads = [read.strip() for read in reads]
        
    # set sequence length
    sequence_length = len(reads[0].strip())

    # iterate through the full sequences
    for read in reads:

        #randomize indices to split the sequence on based on the splitting probability
        splitting_indices = []
        for i in range(sequence_length):
            if random.random() < splitting_probabality:
                splitting_indices.append(i)

        # split the sequence into fragments
        fragments = []
        start = 0
        for index in splitting_indices:
            fragments.append(read[start:index])
            start = index
        fragments.append(read[start:])
        
        # set position that are more then read_length away from the start or end to zero
        read_fragments = []
        for i in range(len(fragments)):
            # skip fragments that are shorter than double read_length
            if len(fragments[i]) < 2*read_length:
                read_fragments.append(fragments[i])
            else:
                read_fragment = fragments[i][:read_length] + '0'*(len(fragments[i]) - 2*read_length) + fragments[i][-read_length:]
                read_fragments.append(read_fragment)

        aligned_fragments = []
        # pad the read fragments with zeros before and after their cutting indices
        for i in range(len(read_fragments)):
            
            # pad first fragment only with zeros at the end
            if i == 0:
                aligned_fragment = read_fragments[i] + '0'*(sequence_length - len(read_fragments[i]))
            # pad last fragment only with zeros at the beginning
            elif i == len(read_fragments) - 1:
                aligned_fragment = '0'*(sequence_length - len(read_fragments[i])) + read_fragments[i]
            # pad all other fragments with zeros at the beginning and end according to their splitting indices
            else:
                # get the cutting indices
                left_index = splitting_indices[i-1]
                right_index = splitting_indices[i]
                aligned_fragment = '0'*(left_index) + read_fragments[i] + '0'*(sequence_length - right_index)
            # if aligned fragment is not all '0' add it to the list of aligned fragments
            if aligned_fragment != '0'*sequence_length:
                aligned_fragments.append(aligned_fragment)

        # write the aligned fragments to the output file
        with open(file_path_output, 'a') as output_file:
            for aligned_fragment in aligned_fragments:
                output_file.write(aligned_fragment + '\n')

    return None

# parse cigar string into binary string where 1 indicates a match and 0 indicates a mismatch
def parse_cigar(cigar_string):
    # get only numbers from cigar string
    numbers = re.findall(r'\d+', cigar_string)
    # get only letters from cigar string
    letters = re.findall(r'[A-Z]', cigar_string)
    # iterate through numbers
    sequence = ''
    for i in range(len(numbers)):
        # if letter is not a valid symbol, set number to 0
        if letters[i] == 'M':
            sequence += '1' * int(numbers[i])
        else:
            sequence += '0' * int(numbers[i])
    return sequence

# parse quality string into binary string where 1 indicates a high quality base and 0 indicates a low quality base
def parse_quality(quality_string):
    valid_symbols = ('?', "@", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I')
    sequence = ''
    for i in range(len(quality_string)):
        if quality_string[i] in valid_symbols:
            sequence += '1'
        else:
            sequence += '0'
    return sequence

# combine cigar and quality strings into one binary string
def parse_cigar_quality(cigar_string, quality_string):
    quality_seq = parse_quality(quality_string)
    cigar_seq = parse_cigar(cigar_string)
    sequence = ''
    for i in range(len(cigar_seq)):
        if cigar_seq[i] == '1' and quality_seq[i] == '1':
            sequence += '1'
        else:
            sequence += '0'
    return sequence

def parse_sequence(sequence, cigar_string, quality_string):
    valid_symbols = ('A', 'C', 'G', 'T')
    checked_seq = parse_cigar_quality(cigar_string, quality_string)
    parsed_sequence = ''
    for i in range(len(checked_seq)):
        if checked_seq[i] == '1' and sequence[i] in valid_symbols:
            parsed_sequence += sequence[i]
        else:
            parsed_sequence += '0'
    return parsed_sequence

def align_fragment(length_sequence, pos, cigar_string, sequence, quality_string):
    parsed_sequence = parse_sequence(sequence, cigar_string, quality_string)
    aligned_sequence = ''
    # add pos - 1 0s to the beginning of the sequence
    aligned_sequence += '0' * (pos - 1)
    # add parsed sequence to the end of the sequence
    aligned_sequence += parsed_sequence
    # add 0s to the end of the sequence
    aligned_sequence += '0' * (length_sequence - len(aligned_sequence))
    return aligned_sequence

# merge aligned sequences
def merge_sequences(left_sequence, right_sequence, length_sequence):
    merged_sequence = ''
    for i in range(length_sequence):
        if left_sequence[i] == right_sequence[i]:
            merged_sequence += left_sequence[i]
        # if left sequence is 0, add right sequence
        elif left_sequence[i] == '0':
            merged_sequence += right_sequence[i]
        # if right sequence is 0, add left sequence
        elif right_sequence[i] == '0':
            merged_sequence += left_sequence[i]
        # if left and right sequences are different, add 0
        else:
            merged_sequence += '0'

    return merged_sequence

def align_reads_experimental(file_path_reads_left: str, file_path_reads_right: str, file_path_output: str, sequence_length: int):

    # read in the left and right reads line by line
    with open(file_path_reads_left, 'r') as reads_file_left:
        reads_left = reads_file_left.readlines()
        # remove the new line character at the end of each line
        reads_left = [read.strip() for read in reads_left]

    with open(file_path_reads_right, 'r') as reads_file_right:
        reads_right = reads_file_right.readlines()
        # remove the new line character at the end of each line
        reads_right = [read.strip() for read in reads_right]

    # remove first 3 lines from both left and right reads
    reads_left = reads_left[3:]
    reads_right = reads_right[3:]

    # separate lines at tab character
    reads_left = [read.split('\t') for read in reads_left]
    reads_right = [read.split('\t') for read in reads_right]

    # check if column 1 of left and right reads match
    for i in range(len(reads_left)):
        if reads_left[i][0] != reads_right[i][0]:
            print('Error: reads are not in the same order')
            return None
        
    # only use columns 2,3,4,5,6,10 and 11
    reads_left = [[read[1], read[2], read[3], read[4], read[5], read[9], read[10]] for read in reads_left]
    reads_right = [[read[1], read[2], read[3], read[4], read[5], read[9], read[10]] for read in reads_right]

    # print(reads_left[0])
    # print(reads_right[0])

    # convert to numpy array of strings
    reads_left = np.array(reads_left, dtype='str')
    reads_right = np.array(reads_right, dtype='str')

    # only keep reads with flag value 0 or 16 in left and flag value 0 or 16 in right
    indices_left = np.where((reads_left[:,0] == '0') | (reads_left[:,0] == '16'))
    indices_right = np.where((reads_right[:,0] == '0') | (reads_right[:,0] == '16'))
    indices = np.intersect1d(indices_left, indices_right)
    reads_left = reads_left[indices]
    reads_right = reads_right[indices]

    # print('Number of reads after flag filtering: ', len(reads_left))

    # only keep reads with MAPQ value >= 70 in left and right
    indices_left = np.where(reads_left[:,3].astype('int') >= 70)
    indices_right = np.where(reads_right[:,3].astype('int') >= 70)
    indices = np.intersect1d(indices_left, indices_right)
    reads_left = reads_left[indices]
    reads_right = reads_right[indices]

    # print('Number of reads after MAPQ filtering: ', len(reads_left))

    # only keep reads with CIGAR strings that do not contain 'I' or 'D' in left and right
    indices = np.where((np.char.find(reads_left[:,4], 'I') == -1) & (np.char.find(reads_left[:,4], 'D') == -1) & (np.char.find(reads_right[:,4], 'I') == -1) & (np.char.find(reads_right[:,4], 'D') == -1))
    reads_left = reads_left[indices]
    reads_right = reads_right[indices]

    # print('Number of reads after Insertion/Deletion filtering: ', len(reads_left))

    # align left and right reads
    # then merge and write to file
    with open(file_path_output, 'w') as output_file:
        for i in range(len(reads_left)):
            # align left and right reads
            aligned_sequence_left = align_fragment(sequence_length, reads_left[i][2].astype(int), reads_left[i][4], reads_left[i][5], reads_left[i][6])
            aligned_sequence_right = align_fragment(sequence_length, reads_right[i][2].astype(int), reads_right[i][4], reads_right[i][5], reads_right[i][6])
            # merge aligned sequences
            merged_sequence = merge_sequences(aligned_sequence_left, aligned_sequence_right, sequence_length)
            # write to file
            output_file.write(merged_sequence + '\n')

    return None




align_reads_experimental('./data/test_files/experimental/left.1.sam', './data/test_files/experimental/right.2.sam', './data/test_files/experimental/aligned_reads.txt', 536)


# align_reads_simulator('./data/test_files/7.txt', './data/test_files/aligned_reads_bound.txt', 1/20, 7, 42)
# align_reads_simulator('./data/test_files/8.txt', './data/test_files/aligned_reads_unbound.txt', 1/20, 7, 42)


def label_reads(file_path_bound: str, file_path_unbound: str, file_path_wildtype: str, 
                file_path_output: str):
    """
    This function to create labels for the aligned reads of either the MIME 
    experiment or MIME Simulator. Every read gets 3 labels: the first is a 
    binary indicator if the read is bound or unbound (1 for bound, 0 for 
    unbound), the second is the length of the fragment (in bp) and the third
    is the number of mutations in the fragment. The labels are added to the
    end of the read and separated by underscores. The labeled reads are then
    written to the output file (file_path_output).

    Args:
        file_path_bound (str): file path to the aligned reads of the bound 
            pool. (aligned_reads_bound.txt)
        file_path_unbound (str): file path to the aligned reads of the unbound
            pool. (aligned_reads_unbound.txt)
        file_path_wildtype (str): file path to the wildtype sequence (.txt 
            file).
        file_path_output (str): file path to the output file. 
            (labeled_reads.txt)
    """

    # Read in the wildtype sequence as a string
    with open(file_path_wildtype, 'r') as wildtype_file:
        wildtype = wildtype_file.read()

    # read in aligned_reads_bound.txt lien by line
    with open(file_path_bound, 'r') as aligned_reads_bound_file:
        bound = aligned_reads_bound_file.readlines()

    # read in aligned_reads_unbound.txt line by line
    with open(file_path_unbound, 'r') as aligned_reads_unbound_file:
        unbound = aligned_reads_unbound_file.readlines()

    # iterate through the bound reads and create labels
    for read in bound:
        # if read is all zeros, raise an error
        if read.strip() == '0'*len(read):
            raise ValueError('Read is all zeros')
        # add the bound label
        read = read.strip() + '_1'
        # get position of the first non-zero character
        positions = [read.find('A'), read.find('C'), read.find('G'), read.find('T')]
        #remove -1 from positions
        positions = tuple(position for position in positions if position != -1)
        start_read = min(positions)
        # get position of the last non-zero character
        positions = [read.rfind('A'), read.rfind('C'), read.rfind('G'), read.rfind('T')]
        positions = tuple(position for position in positions if position != -1)
        end_read = max(positions)
        # add the length label
        read = read + '_' + str(end_read - start_read + 1)
        # check for mutations
        number_mutations = 0
        for position in range(start_read, end_read + 1):
            if read[position] != '0' and read[position] != wildtype[position]:
                number_mutations += 1
        # add the number of mutations label
        read = read + '_' + str(number_mutations) + '\n'

        # write the read to the output file
        with open(file_path_output, 'a') as output_file:
            output_file.write(read)

    # iterate through the unbound reads and create labels
    for read in unbound:
        # if read is all zeros, raise an error
        if read.strip() == '0'*len(read):
            raise ValueError('Read is all zeros')
        # add the unbound label
        read = read.strip() + '_0'
        # get position of the first non-zero character
        positions = [read.find('A'), read.find('C'), read.find('G'), read.find('T')]
        #remove -1 from positions
        positions = tuple(position for position in positions if position != -1)
        start_read = min(positions)
        # get position of the last non-zero character
        positions = [read.rfind('A'), read.rfind('C'), read.rfind('G'), read.rfind('T')]
        positions = tuple(position for position in positions if position != -1)
        end_read = max(positions)
        # add the length label
        read = read + '_' + str(end_read - start_read + 1)
        # check for mutations
        number_mutations = 0
        for position in range(start_read, end_read + 1):
            if read[position] != '0' and read[position] != wildtype[position]:
                number_mutations += 1
        # add the number of mutations label
        read = read + '_' + str(number_mutations) + '\n'

        # write the read to the output file
        with open(file_path_output, 'a') as output_file:
            output_file.write(read)

    return None

# path_to_bound = './data/test_files/aligned_reads_bound.txt'
# path_to_unbound = './data/test_files/aligned_reads_unbound.txt'
# path_to_wildtype = './data/test_files/wildtype.txt'

# label_reads(path_to_bound, path_to_unbound, path_to_wildtype, './data/test_files/labeled_reads.txt')

def filter_reads(file_path_labeled_reads: str, file_path_output: str, min_length: int):
    """
    This function filters the labeled reads. It discards reads that are shorter
    than min_length. Reads with 2 mutations are additionally written to an 
    extra file (file_path_output + 'double_mutant_reads.txt').

    Args:
        file_path_labeled_reads (str): file path to the labeled reads.
            (labeled_reads.txt)
        file_path_output (str): file path to the output file. (filtered_reads.txt)
        min_length (int): minimum length of the reads to be kept.
    """

    # read in the labeled reads line by line
    with open(file_path_labeled_reads, 'r') as labeled_reads_file:
        labeled_reads = labeled_reads_file.readlines()

    # iterate through the labeled reads and filter them
    for read in labeled_reads:
        # if the read is shorter than min_length, discard it
        if int(read.split('_')[-2]) < min_length:
            continue
        # if the read has 2 mutations, write it to the extra file
        if int(read.split('_')[-1]) == 2:
            with open(file_path_output + 'double_mutant_reads.txt', 'a') as output_file:
                output_file.write(read)
        # write the read to the output file
        with open(file_path_output + 'filtered_reads.txt', 'a') as output_file:
            output_file.write(read)

    return None

# filter_reads('./data/test_files/labeled_reads.txt', './data/test_files/', 10)

def one_hot_encode(sequence : str):
    """
    This function one-hot encodes a sequence.

    Args:
        sequence (str): sequence to be one-hot encoded.

    Returns:
        one_hot_sequence (str): one-hot encoded sequence.
    """

    # initialize the one-hot sequence
    one_hot_sequence = ''

    # iterate through the sequence and one-hot encode it
    for character in sequence:
        if character == 'A':
            one_hot_sequence += '1000'
        elif character == 'C':
            one_hot_sequence += '0100'
        elif character == 'G':
            one_hot_sequence += '0010'
        elif character == 'T':
            one_hot_sequence += '0001'
        elif character == '0':
            one_hot_sequence += '0000'
        else:
            raise ValueError('Invalid character in sequence')

    return one_hot_sequence

def parse_single_pool(file_path_filtered_reads : str, file_path_output : str, 
                      protein_concentrations : list, number_protein_concentrations : int):
    """
    This function parses the filtered reads of one pool of the MIME Experiment.
    It takes the filtered reads, adds a one-hot encoding of the protein
    concentrations and one-hot encodes the reads. The reads are then written to
    a file.

    Args:
        file_path_filtered_reads (str): file path to the filtered reads.
            (filtered_reads.txt)
        file_path_output (str): file path to the output file. (parsed_reads.txt)
        protein_concentrations (list): list of protein concentrations. the 
            length of the list is the number of rounds and the values are the
            indices of the protein concentrations.
        number_protein_concentrations (int): number of protein concentrations in
            total.
    """   
    # read in the filtered reads line by line
    with open(file_path_filtered_reads, 'r') as filtered_reads_file:
        filtered_reads = filtered_reads_file.readlines()

    # iterate through the filtered reads and parse them
    for read in filtered_reads:
        # get only part of the string that contains the read without the labels
        sequence = read.split('_')[0]
        # one-hot encode the read
        one_hot_sequence = one_hot_encode(sequence)
        # create the one-hot encoding of the protein concentrations
        protein_concentration_identifier = ''
        for protein_concentration in protein_concentrations:
            protein_concentration_identifier += '0' * protein_concentration + '1' + '0' * (number_protein_concentrations - protein_concentration - 1)
        # get labels
        bound_label = read.split('_')[-3]
        length_label = read.split('_')[-2]
        number_mutations_label = read.split('_')[-1]
        # create output read
        output_read = protein_concentration_identifier + one_hot_sequence + '_' + bound_label + '_' + length_label + '_' + number_mutations_label
        # write the read to the output file
        with open(file_path_output, 'a') as output_file:
            output_file.write(output_read)

    return None

# parse_single_pool('./data/test_files/filtered_reads.txt', './data/test_files/parsed_reads.txt', [0,2], 4)


def concatenate_pools(file_path_pools : str, file_path_output : str, number_protein_concentrations = 4, number_rounds = 2):
    """
    This function concatenates the filtered reads of all pools of the MIME
    Experiment. It takes the filtered reads, adds a one-hot encoding of the
    protein concentrations and one-hot encodes the reads. The reads of all pools
    are then written to a file. The file subsequently gets shuffled.

    The file path of the pools is assumed to be a folder containing all the
    filtered reads of the pools in the following format (for the case of 2 
    rounds with 4 protein concentrations):

    1_1_filtered_reads.txt
    1_2_filtered_reads.txt
    1_3_filtered_reads.txt
    1_4_filtered_reads.txt
    2_1_filtered_reads.txt
    2_2_filtered_reads.txt
    ...

    Args:
        file_path_pools (str): file path to the folder containing the filtered
            reads of all pools.
        file_path_output (str): file path to the output directory.
        number_protein_concentrations (int): number of protein concentrations in
            total.
        number_rounds (int): number of rounds.
    """
    # assert that number_rounds is either 1 or 2
    assert number_rounds == 1 or number_rounds == 2, 'number_rounds must be either 1 or 2'

    # case number_rounds == 1
    if number_rounds == 1:
        for protein_concentration in range(number_protein_concentrations):
            # parse the reads of the pool
            parse_single_pool(file_path_pools + str(protein_concentration + 1) + '_filtered_reads.txt', 
                              file_path_output + str(protein_concentration + 1) + '_parsed_reads.txt', 
                              [protein_concentration], number_protein_concentrations)
            # open parsed reads file
            with open(file_path_output + str(protein_concentration + 1) + '_parsed_reads.txt', 'r') as parsed_reads_file:
                parsed_reads = parsed_reads_file.readlines()
            # write the parsed reads to the output file
            with open(file_path_output + 'parsed_reads.txt', 'a') as output_file:
                for read in parsed_reads:
                    output_file.write(read)
            # delete the parsed reads file
            os.remove(file_path_output + str(protein_concentration + 1) + '_parsed_reads.txt')

            # do the same for the double_mutant_reads
            # parse the reads of the pool
            parse_single_pool(file_path_pools + str(protein_concentration + 1) + '_double_mutant_reads.txt', 
                              file_path_output + str(protein_concentration + 1) + '_dm_parsed_reads.txt', 
                              [protein_concentration], number_protein_concentrations)
            # open parsed reads file
            with open(file_path_output + str(protein_concentration + 1) + '_dm_parsed_reads.txt', 'r') as parsed_reads_file:
                parsed_reads = parsed_reads_file.readlines()
            # write the parsed reads to the output file
            with open(file_path_output + 'dm_parsed_reads.txt', 'a') as output_file:
                for read in parsed_reads:
                    output_file.write(read)
            # delete the parsed reads file
            os.remove(file_path_output + str(protein_concentration + 1) + '_dm_parsed_reads.txt')


    
    # case number_rounds == 2
    if number_rounds == 2:
        for protein_concentration_1 in range(number_protein_concentrations):
            for protein_concentration_2 in range(number_protein_concentrations):
                
                # parse the reads of the pool
                parse_single_pool(file_path_pools + str(protein_concentration_1 + 1) + '_' + str(protein_concentration_2 + 1) + '_filtered_reads.txt', 
                                  file_path_output + str(protein_concentration_1 + 1) + '_' + str(protein_concentration_2 + 1) + '_parsed_reads.txt', 
                                  [protein_concentration_1, protein_concentration_2], number_protein_concentrations)
                # open parsed reads file
                with open(file_path_output + str(protein_concentration_1 + 1) + '_' + str(protein_concentration_2 + 1) + '_parsed_reads.txt', 'r') as parsed_reads_file:
                    parsed_reads = parsed_reads_file.readlines()
                # write the parsed reads to the output file
                with open(file_path_output + 'parsed_reads.txt', 'a') as output_file:
                    for read in parsed_reads:
                        output_file.write(read)
                # delete the parsed reads file
                os.remove(file_path_output + str(protein_concentration_1 + 1) + '_' + str(protein_concentration_2 + 1) + '_parsed_reads.txt')

                # do the same for the double_mutant_reads
                # parse the reads of the pool
                parse_single_pool(file_path_pools + str(protein_concentration_1 + 1) + '_' + str(protein_concentration_2 + 1) + '_double_mutant_reads.txt',
                                    file_path_output + str(protein_concentration_1 + 1) + '_' + str(protein_concentration_2 + 1) + '_dm_parsed_reads.txt',
                                    [protein_concentration_1, protein_concentration_2], number_protein_concentrations)
                # open parsed reads file
                with open(file_path_output + str(protein_concentration_1 + 1) + '_' + str(protein_concentration_2 + 1) + '_dm_parsed_reads.txt', 'r') as parsed_reads_file:
                    parsed_reads = parsed_reads_file.readlines()
                # write the parsed reads to the output file
                with open(file_path_output + 'dm_parsed_reads.txt', 'a') as output_file:
                    for read in parsed_reads:
                        output_file.write(read)
                # delete the parsed reads file
                os.remove(file_path_output + str(protein_concentration_1 + 1) + '_' + str(protein_concentration_2 + 1) + '_dm_parsed_reads.txt')
                

    # shuffle the output file
    with open(file_path_output + 'parsed_reads.txt', 'r') as output_file:
        parsed_reads = output_file.readlines()
    random.shuffle(parsed_reads)
    with open(file_path_output + 'parsed_reads.txt', 'w') as output_file:
        for read in parsed_reads:
            output_file.write(read)

    # shuffle the output file
    with open(file_path_output + 'dm_parsed_reads.txt', 'r') as output_file:
        parsed_reads = output_file.readlines()
    random.shuffle(parsed_reads)
    with open(file_path_output + 'dm_parsed_reads.txt', 'w') as output_file:
        for read in parsed_reads:
            output_file.write(read)
    
    return None

# concatenate_pools('./data/test_files/pooled/', './data/test_files/', 2, 2)

def parse_simulation(file_path_simulation_output : str, file_path_output : str, file_path_wildtype : str,
                     protein_concentrations : list, number_rounds = 2,
                     min_length = 30, splitting_probability : float  = 3/100, read_length = 23,
                     seed = 42):
    """
    This function parses the output of the MIME simulator. It takes the 
    simulators output as produced by the sim_automation.py script and outputs 
    training data sets for the MIME model as well as the distribution of the 
    read lengths and mutations per read. The output files are written to the
    specified file path.

    Args:
        file_path_simulation_output (str): file path to the output directory of 
            the MIME simulator (from sim_automation.py).
        file_path_output (str): file path to the output directory.
        file_path_wildtype (str): file path to the wildtype sequence.
        protein_concentrations (list): list of protein concentrations used in
            the simulation.
        number_rounds (int, optional): number of rounds performed. Defaults to 
            2.
        min_length (int, optional): the minimum read length to be kept 
            (everything lower will be filtered out). Defaults to 30.
        splitting_probability (float, optional): probability for a mutation at 
            a given position. Defaults to 3/100.
        read_length (int, optional): the length with the fragments are read in 
            (from each side). Defaults to 23.
        seed (int, optional): seed for the random number generator (used for 
            random fragmentation). Defaults to 42.
    """

    # assert that number_rounds is either 1 or 2
    assert number_rounds == 1 or number_rounds == 2, 'number_rounds must be either 1 or 2'

    # case number_rounds == 1
    if number_rounds == 1:

        #iterate over all protein concentrations
        for i in tqdm(range(len(protein_concentrations))):

            protein_concentration = protein_concentrations[i]
            
            # set file path for the current protein concentration
            file_path = file_path_simulation_output + 'prot' + str(protein_concentration) + '/sequences/'

            # align the reads
            align_reads_simulator(
                file_path_reads=file_path + '3.txt',
                file_path_output='./data/cache/aligned_reads_bound.txt',
                splitting_probabality=splitting_probability,
                read_length=read_length,
                seed=seed
            )
            align_reads_simulator(
                file_path_reads=file_path + '4.txt',
                file_path_output='./data/cache/aligned_reads_unbound.txt',
                splitting_probabality=splitting_probability,
                read_length=read_length,
                seed=seed**2
            )

            # label the reads
            label_reads(
                file_path_bound='./data/cache/aligned_reads_bound.txt',
                file_path_unbound='./data/cache/aligned_reads_unbound.txt',
                file_path_wildtype=file_path_wildtype,
                file_path_output='./data/cache/labelled_reads.txt'
            )

            # read the labelled reads
            with open('./data/cache/labelled_reads.txt', 'r') as labelled_reads_file:
                labelled_reads = labelled_reads_file.readlines()

            # get the read lengths and the number of mutations per read
            read_lengths = []
            mutations_per_read_bound = []
            mutations_per_read_unbound = []
            for read in labelled_reads:
                # labels are separated by an underscore
                labels = read.split('_')
                # the second label is the read length
                read_lengths.append(int(labels[2]))
                # the third label is the number of mutations
                if labels[1] == '1':
                    mutations_per_read_bound.append(int(labels[3]))
                elif labels[1] == '0':
                    mutations_per_read_unbound.append(int(labels[3]))
                else:
                    raise ValueError('Invalid bound/unbound label')
                
            # write the read lengths and the number of mutations per read to the output directory
            with open(file_path_output + str(i+1) + '_read_lengths.txt', 'w') as output_file:
                for read_length in read_lengths:
                    output_file.write(str(read_length) + ' ')
            with open(file_path_output + str(i+1) + '_mpr_bound.txt', 'w') as output_file:
                for mutations in mutations_per_read_bound:
                    output_file.write(str(mutations) + ' ')
            with open(file_path_output + str(i+1) + '_mpr_unbound.txt', 'w') as output_file:
                for mutations in mutations_per_read_unbound:
                    output_file.write(str(mutations) + ' ')

            # delete the aligned reads
            os.remove('./data/cache/aligned_reads_bound.txt')
            os.remove('./data/cache/aligned_reads_unbound.txt')

            # filter the reads
            filter_reads(
                file_path_labeled_reads='./data/cache/labelled_reads.txt',
                file_path_output='./data/cache/pooled/',
                min_length=min_length
            )

            # get the number of mutations per read for the filtered reads
            with open('./data/cache/pooled/filtered_reads.txt', 'r') as filtered_reads_file:
                filtered_reads = filtered_reads_file.readlines()

            mutations_per_read_bound = []
            mutations_per_read_unbound = []
            for read in filtered_reads:
                # labels are separated by an underscore
                labels = read.split('_')
                # the third label is the number of mutations
                if labels[1] == '1':
                    mutations_per_read_bound.append(int(labels[3]))
                elif labels[1] == '0':
                    mutations_per_read_unbound.append(int(labels[3]))
                else:
                    raise ValueError('Invalid bound/unbound label')

            # write the number of mutations per read to the output directory
            with open(file_path_output + str(i+1) + '_mprf_bound.txt', 'w') as output_file:
                for mutations in mutations_per_read_bound:
                    output_file.write(str(mutations) + ' ')
            with open(file_path_output + str(i+1) + '_mprf_unbound.txt', 'w') as output_file:
                for mutations in mutations_per_read_unbound:
                    output_file.write(str(mutations) + ' ')

            # delete the labelled reads
            os.remove('./data/cache/labelled_reads.txt')

            # rename filtered_reads.txt and double_mutant_reads.txt according to the protein concentration
            os.rename('./data/cache/pooled/filtered_reads.txt', './data/cache/pooled/' + str(i+1) + '_filtered_reads.txt')
            os.rename('./data/cache/pooled/double_mutant_reads.txt', './data/cache/pooled/' + str(i+1) + '_double_mutant_reads.txt')

        # concatenate the pools

        concatenate_pools(
            file_path_pools='./data/cache/pooled/',
            file_path_output=file_path_output,
            number_protein_concentrations=len(protein_concentrations),
            number_rounds=number_rounds
        )

        # shuffle parsed_reads.txt and dm_parsed_reads.txt
        with open(file_path_output + 'parsed_reads.txt', 'r') as output_file:
            parsed_reads = output_file.readlines()
        random.shuffle(parsed_reads)
        with open(file_path_output + 'parsed_reads.txt', 'w') as output_file:
            for read in parsed_reads:
                output_file.write(read)

        with open(file_path_output + 'dm_parsed_reads.txt', 'r') as output_file:
            parsed_reads = output_file.readlines()
        random.shuffle(parsed_reads)
        with open(file_path_output + 'dm_parsed_reads.txt', 'w') as output_file:
            for read in parsed_reads:
                output_file.write(read)

        # delete the pools
        shutil.rmtree('./data/cache/pooled/')

        # create the pooled directory again
        os.mkdir('./data/cache/pooled/')

    # case number_rounds == 2
    if number_rounds == 2:

        # iterate over all protein concentrations
        for i in tqdm(range(len(protein_concentrations))):
            for j in tqdm(range(len(protein_concentrations)), leave=False):

                first_protein_concentration = protein_concentrations[i]
                second_protein_concentration = protein_concentrations[j]

                # set the file path for current protein concentration pair
                file_path = file_path_simulation_output + 'secondFromProt' + str(first_protein_concentration) + '/prot' + str(second_protein_concentration) + '/sequences/'


                # align the reads
                align_reads_simulator(
                    file_path_reads=file_path + '7.txt',
                    file_path_output='./data/cache/aligned_reads_bound.txt',
                    splitting_probabality=splitting_probability,
                    read_length=read_length,
                    seed=seed
                )
                align_reads_simulator(
                    file_path_reads=file_path + '8.txt',
                    file_path_output='./data/cache/aligned_reads_unbound.txt',
                    splitting_probabality=splitting_probability,
                    read_length=read_length,
                    seed=seed**2
                )

                # label the reads
                label_reads(
                    file_path_bound='./data/cache/aligned_reads_bound.txt',
                    file_path_unbound='./data/cache/aligned_reads_unbound.txt',
                    file_path_wildtype=file_path_wildtype,
                    file_path_output='./data/cache/labelled_reads.txt'
                )

                # read the labelled reads
                with open('./data/cache/labelled_reads.txt', 'r') as labelled_reads_file:
                    labelled_reads = labelled_reads_file.readlines()

                # get the read lengths and the number of mutations per read
                read_lengths = []
                mutations_per_read_bound = []
                mutations_per_read_unbound = []
                for read in labelled_reads:
                    # labels are separated by an underscore
                    labels = read.split('_')
                    # the second label is the read length
                    read_lengths.append(int(labels[2]))
                    # the third label is the number of mutations
                    if labels[1] == '1':
                        mutations_per_read_bound.append(int(labels[3]))
                    elif labels[1] == '0':
                        mutations_per_read_unbound.append(int(labels[3]))
                    else:
                        raise ValueError('Invalid bound/unbound label')

                # write the read lengths and the number of mutations per read to the output directory
                with open(file_path_output + str(i+1) + '_' + str(j+1) + '_read_lengths.txt', 'w') as output_file:
                    for read_length in read_lengths:
                        output_file.write(str(read_length) + ' ')
                with open(file_path_output + str(i+1) + '_' + str(j+1) + '_mpr_bound.txt', 'w') as output_file:
                    for mutations in mutations_per_read_bound:
                        output_file.write(str(mutations) + ' ')
                with open(file_path_output + str(i+1) + '_' + str(j+1) + '_mpr_unbound.txt', 'w') as output_file:
                    for mutations in mutations_per_read_unbound:
                        output_file.write(str(mutations) + ' ')

                # delete the aligned reads
                os.remove('./data/cache/aligned_reads_bound.txt')
                os.remove('./data/cache/aligned_reads_unbound.txt')

                # filter the reads
                filter_reads(
                    file_path_labeled_reads='./data/cache/labelled_reads.txt',
                    file_path_output='./data/cache/pooled/',
                    min_length=min_length
                )

                # get the number of mutations per read for the filtered reads
                with open('./data/cache/pooled/filtered_reads.txt', 'r') as filtered_reads_file:
                    filtered_reads = filtered_reads_file.readlines()

                mutations_per_read_bound = []
                mutations_per_read_unbound = []
                for read in filtered_reads:
                    # labels are separated by an underscore
                    labels = read.split('_')
                    # the third label is the number of mutations
                    if labels[1] == '1':
                        mutations_per_read_bound.append(int(labels[3]))
                    elif labels[1] == '0':
                        mutations_per_read_unbound.append(int(labels[3]))
                    else:
                        raise ValueError('Invalid bound/unbound label')

                # write the number of mutations per read to the output directory
                with open(file_path_output + str(i+1) + '_' + str(j+1) + '_mprf_bound.txt', 'w') as output_file:
                    for mutations in mutations_per_read_bound:
                        output_file.write(str(mutations) + ' ')
                with open(file_path_output + str(i+1) + '_' + str(j+1) + '_mprf_unbound.txt', 'w') as output_file:
                    for mutations in mutations_per_read_unbound:
                        output_file.write(str(mutations) + ' ')

                # delete the labelled reads
                os.remove('./data/cache/labelled_reads.txt')

                # rename filtered_reads.txt and double_mutant_reads.txt according to the protein concentration
                os.rename('./data/cache/pooled/filtered_reads.txt', './data/cache/pooled/' + str(i+1) + '_' + str(j+1) + '_filtered_reads.txt')
                os.rename('./data/cache/pooled/double_mutant_reads.txt', './data/cache/pooled/' + str(i+1) + '_' + str(j+1) + '_double_mutant_reads.txt')

        # concatenate the pools

        concatenate_pools(
            file_path_pools='./data/cache/pooled/',
            file_path_output=file_path_output,
            number_protein_concentrations=len(protein_concentrations),
            number_rounds=number_rounds
        )

        # shuffle parsed_reads.txt and dm_parsed_reads.txt
        with open(file_path_output + 'parsed_reads.txt', 'r') as output_file:
            parsed_reads = output_file.readlines()
        random.shuffle(parsed_reads)
        with open(file_path_output + 'parsed_reads.txt', 'w') as output_file:
            for read in parsed_reads:
                output_file.write(read)

        with open(file_path_output + 'dm_parsed_reads.txt', 'r') as output_file:
            parsed_reads = output_file.readlines()
        random.shuffle(parsed_reads)
        with open(file_path_output + 'dm_parsed_reads.txt', 'w') as output_file:
            for read in parsed_reads:
                output_file.write(read)

        # delete the pools
        shutil.rmtree('./data/cache/pooled/')

        # create the pooled directory again
        os.mkdir('./data/cache/pooled/')




    return None

# # parse first round of simulation data
# parse_simulation(
#     file_path_simulation_output='./data/simulation_data/experimental_conditions_2/',
#     file_path_output='./data/simulation_data/experimental_conditions_2/first_round/',
#     file_path_wildtype='./data/simulation_data/wildtype.txt',
#     protein_concentrations=[.1, 1, 10],
#     number_rounds=1,
#     min_length=30,
#     splitting_probability=3/100,
#     read_length=23,
#     seed=42
# )

# # parse second round of simulation data
# parse_simulation(
#     file_path_simulation_output='./data/simulation_data/experimental_conditions_2/',
#     file_path_output='./data/simulation_data/experimental_conditions_2/second_round/',
#     file_path_wildtype='./data/simulation_data/wildtype.txt',
#     protein_concentrations=[.1, 1, 10],
#     number_rounds=2,
#     min_length=30,
#     splitting_probability=3/100,
#     read_length=23,
#     seed=24
# )