#run with "pythonw" as framework
'''
a collection of codes run analysis on glass structures
python3 code
need to install ase first

'''
import numpy as np
from ase.io import read
from ase import neighborlist, geometry
import formatxfer as fx
import matplotlib.pyplot as plt

def get_rdf_all(atoms, rMax, nBins = 200):
    #calculate pair distribution function
    d = neighborlist.neighbor_list('d', atoms, rMax)
    dr = rMax/nBins
    edges = np.arange(0., rMax + 1.1 *dr, dr)
    h, binEdges = np.histogram(d, edges)
    rho = len(atoms) / atoms.get_volume() 
    factor = 4./3. * np.pi * rho * len(atoms)
    rdf = h / (factor * (binEdges[1:]**3 - binEdges[:-1]**3)) 

    plt.plot(binEdges[1:], rdf)
    plt.savefig('rdfAll.pdf')
    plt.close()
    peaks = (np.diff(np.sign(np.diff(rdf))) < 0).nonzero()[0] + 1 # local max
    firstPeakInd = np.argmax(rdf[peaks])
    firstPeak = binEdges[peaks[firstPeakInd]]
    #peaks_2 = np.delete(peaks, firstPeakInd)
    #secondPeakInd = np.argmax(rdf[peaks_2])
    #secondPeak = binEdges[peaks_2[secondPeakInd]]
    print("first peak of rdf: %12.8f" % firstPeak)
    cutoff = firstPeak*1.25
    print("first NN cutoff set to: %12.8f" % cutoff)
    return(cutoff)

def get_NL(atoms, cutoff):
    #get neighbor list
    NBList = neighborlist.neighbor_list('ijDd', atoms, cutoff)
    nnn = np.bincount(NBList[0]) #number of nearesr neighbors
    return(nnn, NBList[0], NBList[1], NBList[2], NBList[3])

def get_rdf_A_B(types, atoms, inFile, rMax, nBins = 200):

    with open(inFile,'r') as fin:
        lines=fin.readlines()
        listTypeName=[str(s) for s in lines[5].split()]
        listTypeNum=[int(s) for s in lines[6].split()]
    print(listTypeName)
    print(listTypeNum)

    #get index of element type of interest
    typeIndex=[]
    for iType in types:
        for j in range(len(listTypeName)):
            if iType == listTypeName[j]:
                typeIndex.append(j)
    typeAStart = 0
    typeAEnd = 0
    for i in range(len(listTypeNum)):
        if i < typeIndex[0]:
            typeAStart += listTypeNum[i]
    typeAEnd = typeAStart + listTypeNum[typeIndex[0]]
    print(types[0],typeAStart,typeAEnd)
    typeBStart = 0
    typeBEnd = 0
    for i in range(len(listTypeNum)):
        if i < typeIndex[1]:
            typeBStart += listTypeNum[i]
    typeBEnd = typeBStart + listTypeNum[typeIndex[1]]
    print(types[1],typeBStart,typeBEnd)

    atoms_A = atoms[typeAStart:typeAEnd]
    atoms_B = atoms[typeBStart:typeBEnd]
    print(len(atoms_A))
    print(len(atoms_B))
    atoms_new = atoms_A + atoms_B
    d = neighborlist.neighbor_list('d', atoms_new, rMax)
    dr = rMax/nBins
    edges = np.arange(0., rMax + 1.1 *dr, dr)
    h, binEdges = np.histogram(d, edges)
    rho = len(atoms_new) / atoms.get_volume() 
    factor = 4./3. * np.pi * rho * len(atoms_new)
    rdf = h / (factor * (binEdges[1:]**3 - binEdges[:-1]**3)) 

    plt.plot(binEdges[1:], rdf)
    plt.savefig('rdf'+types[0]+'_'+types[1]+'.pdf')
    plt.close()


    return






inFile = 'POSCAR'
atoms = read(inFile)
rMax = 10.0
cutoff = get_rdf_all(atoms, rMax)
#NBList = get_NL(atoms, cutoff)
types = ['Si','O']
get_rdf_A_B(types, atoms, inFile, rMax)
types = ['Ca','O']
get_rdf_A_B(types, atoms, inFile, rMax)
types = ['Al','O']
get_rdf_A_B(types, atoms, inFile, rMax)
