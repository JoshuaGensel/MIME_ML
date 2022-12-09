import subprocess
import os
from random import *

pathToSim = "~/programming_projects/dmMIMESim/build/bin/MIMESim_prog"
workingDirectory = "/mnt/d/data/MIME_data/simData/dmMIME/seqLen100/fullEpistasis"

#parameters
L = 100
q = 4
M = 100000
p_mut = 0.0225
p_error = 0.0045
p_effect = 1
p_epistasis = 1
epi_restrict = 0
proteinConcentrations = [1, 6, 15, 30]

#create output directories
os.mkdir(workingDirectory)
firstRoundDirectories = []
secondRoundDirectories = []
secondRoundSubDirectories = []
for proteinConcentration in proteinConcentrations:
    #create first round directories
    firstDir = workingDirectory + "/prot" + str(proteinConcentration)
    os.mkdir(firstDir)
    firstRoundDirectories.append(firstDir)
    
    #write parameters.txt file for first round
    paramsFile = firstDir + "/parameters.txt"
    with open(paramsFile, 'w') as f:
        f.write("L\t" + str(L) + "\n")
        f.write("q\t" + str(q) + "\n")
        f.write("M\t" + str(M) + "\n")
        f.write("p_mut\t" + str(p_mut) + "\n")
        f.write("p_error\t" + str(p_error) + "\n")
        f.write("B_tot\t" + str(proteinConcentration) + "\n")
        f.write("seed\t" + str(randint(0, 999999)) + "\n")
        f.write("p_epistasis\t" + str(p_epistasis) + "\n")
        f.write("p_effect\t" + str(p_effect) + "\n")
        f.write("epi_restrict\t" + str(epi_restrict) + "\n")
    
    #create second round directories
    secondDir = workingDirectory + "/secondFromProt" + str(proteinConcentration)
    os.mkdir(secondDir)
    secondRoundDirectories.append(secondDir)
    #create second round subdirectories
    for proteinConcentration2 in proteinConcentrations:
        subDir = secondDir + "/prot" + str(proteinConcentration2)
        os.mkdir(subDir)
        secondRoundSubDirectories.append(subDir)
        
        #write parameters.txt file for second round
        paramsFile = subDir + "/parameters.txt"
        with open(paramsFile, 'w') as f:
            f.write("L\t" + str(L) + "\n")
            f.write("q\t" + str(q) + "\n")
            f.write("M\t" + str(M) + "\n")
            f.write("p_mut\t" + str(p_mut) + "\n")
            f.write("p_error\t" + str(p_error) + "\n")
            f.write("B_tot\t" + str(proteinConcentration2) + "\n")
            f.write("seed\t" + str(randint(0, 999999)) + "\n")
            f.write("p_epistasis\t" + str(p_epistasis) + "\n")
            f.write("p_effect\t" + str(p_effect) + "\n")
            f.write("epi_restrict\t" + str(epi_restrict) + "\n")

#run first simulation
command = pathToSim + " --working-dir " + firstRoundDirectories[0]
subprocess.call(command, shell=True)

#copy single_kds.txt and pairwise_epistasis.txt to all other directories
for i in range(1, len(firstRoundDirectories)):
    command = "cp " + firstRoundDirectories[0] + "/single_kds.txt " + firstRoundDirectories[i]
    subprocess.call(command, shell=True)
    command = "cp " + firstRoundDirectories[0] + "/pairwise_epistasis.txt " + firstRoundDirectories[i]
    subprocess.call(command, shell=True)
    
    #run simulations for rest of first round
    command = pathToSim + " --working-dir " + firstRoundDirectories[i] + " --read-gt"
    subprocess.call(command, shell=True)
    
#run second round simulations
for i in range(0, len(secondRoundSubDirectories)):
    #copy single_kds.txt and pairwise_epistasis.txt to all other directories
    command = "cp " + firstRoundDirectories[0] + "/single_kds.txt " + secondRoundSubDirectories[i]
    subprocess.call(command, shell=True)
    command = "cp " + firstRoundDirectories[0] + "/pairwise_epistasis.txt " + secondRoundSubDirectories[i]
    subprocess.call(command, shell=True)
    
for i in range(0, len(secondRoundSubDirectories)):
    for pC in proteinConcentrations:
        if f'FromProt{pC}/' in secondRoundSubDirectories[i]:
            command = pathToSim + " --working-dir " + secondRoundSubDirectories[i] + " --previous-dir " + firstRoundDirectories[proteinConcentrations.index(pC)]
            subprocess.call(command, shell=True)