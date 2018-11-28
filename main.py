'''
Program header:
    Mingfei Zhang
run with "pythonw" as framework
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

def get_rdf_A_B(types, atoms, listTypeName, listTypeNum, rMax, nBins = 200):
    #get index of element types of interest
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
    atoms_new = atoms_A + atoms_B
    d = neighborlist.neighbor_list('d', atoms_new, rMax)
    dr = rMax/nBins
    edges = np.arange(0., rMax + 1.1 *dr, dr)
    h, binEdges = np.histogram(d, edges)
    rho = len(atoms_new) / atoms.get_volume() 
    factor = 4./3. * np.pi * rho * len(atoms_new)
    rdf = h / (factor * (binEdges[1:]**3 - binEdges[:-1]**3)) 

    plt.plot(binEdges[1:], rdf)
    plt.savefig('rdf_'+types[0]+'_'+types[1]+'.pdf')
    plt.close()

    peaks = (np.diff(np.sign(np.diff(rdf))) < 0).nonzero()[0] + 1 # local max
    firstPeakInd = np.argmax(rdf[peaks])
    firstPeak = binEdges[peaks[firstPeakInd]]
    #peaks_2 = np.delete(peaks, firstPeakInd)
    #secondPeakInd = np.argmax(rdf[peaks_2])
    #secondPeak = binEdges[peaks_2[secondPeakInd]]
    print("first peak of rdf: %12.8f" % firstPeak)
    #print("second peak of rdf: %12.8f" % secondPeak)
    #cutoff = (firstPeak + secondPeak)/2.0
    cutoff = firstPeak*1.25
    print("first NN cutoff set to: %12.8f" % cutoff)

    #calculate CN
    NBList = neighborlist.neighbor_list('ij', atoms_new, cutoff)
    nnn = np.bincount(NBList[0]) #number of nearesr neighbors
    typeA_CN = np.mean(nnn[:len(atoms_A)])
    typeB_CN = np.mean(nnn[len(atoms_A):])

    print("CN of %s : %8.6f" % (types[0], typeA_CN))
    print("CN of %s : %8.6f" % (types[1], typeB_CN))
    return

def get_adf_O_M_O(MType, atoms, listTypeName, listTypeNum, rMax, cutoff, dr = 1.0):
    #get index of element types of interest
    typeIndex=[]
    for j in range(len(listTypeName)):
        if MType == listTypeName[j]:
            typeIndex.append(j)
    MTypeStart = 0
    MTypeEnd = 0
    for i in range(len(listTypeNum)):
        if i < typeIndex[0]:
            MTypeStart += listTypeNum[i]
    MTypeEnd = MTypeStart + listTypeNum[typeIndex[0]]
    print(MType, MTypeStart, MTypeEnd)
    
    OType = 'O'
    typeIndex=[]
    for j in range(len(listTypeName)):
        if 'O' == listTypeName[j]:
            typeIndex.append(j)
    OTypeStart = 0
    OTypeEnd = 0
    for i in range(len(listTypeNum)):
        if i < typeIndex[0]:
            OTypeStart += listTypeNum[i]
    OTypeEnd = OTypeStart + listTypeNum[typeIndex[0]]
    print(OType, OTypeStart, OTypeEnd)

    NBList = neighborlist.neighbor_list('ijDd', atoms, cutoff)
    nnn = np.bincount(NBList[0]) #number of nearesr neighbors
    # M as i
    MIndexList = ((NBList[0] >= MTypeStart) & (NBList[0] < MTypeEnd)).nonzero()[0]
    # O as j
    OIndexList = ((NBList[1] >= OTypeStart) & (NBList[1] < OTypeEnd)).nonzero()[0]
    print(nnn)
    print(len(MIndexList))
    angles = []
    for i in range(MTypeStart, MTypeEnd):
        iNNVector = []
        if nnn[i] < 2:
            continue
        else:
            MOIndex = ((NBList[0] == i) & (NBList[1] < OTypeEnd) &\
                       (NBList[1] >= OTypeStart)).nonzero()[0]
            for j in range(len(MOIndex)):
                for k in range(j+1,len(MOIndex)+1):
                    angles.append(atoms.get_angle(NBList[1][j], NBList[0][i],\
                                  NBList[1][k], mic=True))
    nBins = 180. / dr
    edges = np.arange(0.1, 180.0, dr)
    h, binEdges = np.histogram(angles, edges)

    plt.plot(binEdges[1:], h)
    plt.savefig('adf_All.pdf')
    plt.close()

    return

def processAll(inFile):
    print("working on: %s" % inFile)
    with open(inFile,'r') as fin:
        lines=fin.readlines()
        listTypeName=[str(s) for s in lines[5].split()]
        listTypeNum=[int(s) for s in lines[6].split()]
    atoms = read(inFile)
    rMax = 10.0
    #cutoff = get_rdf_all(atoms, rMax) 
    ##NBList = get_NL(atoms, cutoff)
    #types = ['Si','O']
    #get_rdf_A_B(types, atoms, listTypeName, listTypeNum, rMax)
    #types = ['Ca','O']
    #get_rdf_A_B(types, atoms, listTypeName, listTypeNum, rMax)
    #types = ['Al','O']
    #get_rdf_A_B(types, atoms, listTypeName, listTypeNum, rMax)
    cutoff = 2.25
    dr = 2.0
    get_adf_O_M_O('Si', atoms, listTypeName, listTypeNum, rMax, cutoff, dr)
    return

inFile = 'POSCAR'
processAll(inFile)

