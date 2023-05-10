import numpy as np
from tqdm import tqdm
import os 
import random

def label_reads(file_path_bound: str, file_path_unbound: str, file_path_wildtype: str, file_path_output: str):
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
        if -1 in positions: 
            positions.remove(-1)
        start_read = min(positions) 
        # get position of the last non-zero character
        positions = [read.rfind('A'), read.rfind('C'), read.rfind('G'), read.rfind('T')]
        if -1 in positions:
            positions.remove(-1)
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
        if -1 in positions: 
            positions.remove(-1)
        start_read = min(positions)
        # get position of the last non-zero character
        positions = [read.rfind('A'), read.rfind('C'), read.rfind('G'), read.rfind('T')]
        if -1 in positions:
            positions.remove(-1)
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


# test the functions

path_to_bound = './src/expFiles/fragmentedDataBound.txt'
path_to_unbound = './src/expFiles/fragmentedDataUnbound.txt'
path_to_wildtype = './src/expFiles/wildtype.txt'

label_reads(path_to_bound, path_to_unbound, path_to_wildtype, './src/expFiles/labeled_reads.txt')
filter_reads('./src/expFiles/labeled_reads.txt', './src/expFiles/', 10)