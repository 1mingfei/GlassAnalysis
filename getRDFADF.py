#!/usr/bin/env pythonw
'''
Program header:
    Mingfei Zhang
run with "pythonw" as framework
a collection of codes run analysis on glass structures
python3 code
need to install ase first

'''
import os
import sys
import numpy as np
from ase.io import read
from ase import neighborlist, geometry
import matplotlib.pyplot as plt

def get_rdf_all(atoms, rMax, nBins = 200):
    #calculate pair distribution function
    d = neighborlist.neighbor_list('d', atoms, rMax)
    dr = rMax/nBins
    edges = np.arange(0., rMax + 1.1 *dr, dr)
    h, binEdges = np.histogram(d, edges)
    rho = len(atoms) / atoms.get_volume() 
    factor = 4./ 3. * np.pi * rho * len(atoms)
    rdf = h / (factor * (binEdges[1:]**3 - binEdges[:-1]**3)) 

    plt.plot(binEdges[1:], rdf)
    plt.savefig('RDFADF/rdfAll.pdf')
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

def get_rdf_M_O(MType, atoms, listTypeName, listTypeNum, rMax, nBins = 200):
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
    #print(MType, MTypeStart, MTypeEnd)
    
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
    #print(OType, OTypeStart, OTypeEnd)

    atoms_M = atoms[MTypeStart:MTypeEnd]
    MLength = len(atoms_M)
    atoms_O = atoms[OTypeStart:OTypeEnd]
    atoms_new = atoms_M + atoms_O
    listA, listB, d = neighborlist.neighbor_list('ijd', atoms_new, rMax)
    listMO_indexM=set()
    listMO_indexO=set()
    d_MO = []
    for i in range(len(d)):
        if ((listA[i] < MLength) \
            and (listB[i] >= MLength)):
            d_MO.append(d[i])
            listMO_indexM.add(listA[i])
            listMO_indexO.add(listB[i])
    dr = rMax/nBins
    edges = np.arange(0., rMax + 1.1 * dr, dr)
    h, binEdges = np.histogram(d_MO, edges)
    #rho = len(atoms_new) / atoms.get_volume() 
    #factor = 4. / 3. * np.pi * rho * len(atoms_new)

    rho = 0.5*(len(listMO_indexM)+len(listMO_indexO)) / atoms.get_volume()
    factor = 4./3. * np.pi * rho * 0.5*(len(listMO_indexM)+len(listMO_indexO))


    rdf = h / (factor * (binEdges[1:]**3 - binEdges[:-1]**3)) 

    plt.plot(binEdges[1:], rdf)
    plt.savefig('RDFADF/rdf_' + MType + '_O.pdf')
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
    cutoff = firstPeak * 1.25
    print("first NN cutoff set to: %12.8f" % cutoff)

    #calculate CN
    NBList = neighborlist.neighbor_list('ij', atoms_new, cutoff)
    d = neighborlist.neighbor_list('d', atoms_new, cutoff)
    nnn = np.bincount(NBList[0]) #number of nearesr neighbors
    typeM_CN = np.mean(nnn[:len(atoms_M)])
    typeO_CN = np.mean(nnn[len(atoms_M):])

    print("CN of %s : %8.6f" % (MType, typeM_CN))
    print("CN of %s : %8.6f" % ('O', typeO_CN))
    aveBondD = np.mean(d)
    stdBondD = np.std(d)
    print("averaged bond distance : %12.8f" % aveBondD)
    print("bond distance std : %12.8f" % stdBondD)
    BOIndexList, NBOIndexList  = get_BOList(listTypeName, listTypeNum, atoms,\
                                            cutoff)[0],\
                                 get_BOList(listTypeName, listTypeNum, atoms,\
                                            cutoff)[1]
    #M-BO bond distance
    print("%s and bridge O bond distribution" % MType)
    print("number of bridge O within cutoff(%12.8fA): %d"%(cutoff, len(BOIndexList)))
    print("number of nonbridge O within cutoff(%12.8fA): %d"%(cutoff, len(NBOIndexList)))
    atoms_new = atoms_M
    MLength = len(atoms_M)
    for i in (BOIndexList):
        atoms_new += atoms[i]
    listA, listB, d = neighborlist.neighbor_list('ijd', atoms_new, rMax)
    listMBO_indexM=set()
    listMBO_indexO=set()
    d_MO = []
    for i in range(len(d)):
        if ((listA[i] < MLength) \
            and (listB[i] >= MLength)):
            d_MO.append(d[i])
            listMBO_indexM.add(listA[i])
            listMBO_indexO.add(listB[i])

    dr = rMax/nBins
    edges = np.arange(0., rMax + 1.1 *dr, dr)
    h, binEdges = np.histogram(d_MO, edges)
    #rho = len(atoms_new) / atoms.get_volume()
    #factor = 4./3. * np.pi * rho * len(atoms_new)
    rho = 0.5*(len(listMBO_indexM)+len(listMBO_indexO)) / atoms.get_volume()
    factor = 4./3. * np.pi * rho * 0.5*(len(listMBO_indexM)+len(listMBO_indexO))

    rdf = h / (factor * (binEdges[1:]**3 - binEdges[:-1]**3))

    plt.plot(binEdges[1:], rdf)
    plt.savefig('RDFADF/rdf_'+MType+'_BO.pdf')
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
    cutoff_BO = firstPeak*1.25
    print("first NN cutoff set to: %12.8f" % cutoff)

    #calculate CN
    NBList = neighborlist.neighbor_list('ij', atoms_new, cutoff)
    d = neighborlist.neighbor_list('d', atoms_new, cutoff_BO)
    nnn = np.bincount(NBList[0]) #number of nearesr neighbors
    typeM_CN = np.mean(nnn[:MLength])
    typeO_CN = np.mean(nnn[MLength:])

    print("CN of %s : %8.6f" % (MType, typeM_CN))
    print("CN of %s : %8.6f" % ('BO', typeO_CN))
    aveBondD = np.mean(d)
    stdBondD = np.std(d)
    print("averaged bond distance : %8.6f" % aveBondD)
    print("bond distance std : %8.6f" % stdBondD)

    #M-NBO bond distance
    print("%s and non bridge O bond distribution" % MType)
    atoms_M = atoms[MTypeStart:MTypeEnd]
    atoms_new = atoms_M
    MLength = len(atoms_M)
    for i in (NBOIndexList):
        atoms_new += atoms[i]
    #d = neighborlist.neighbor_list('d', atoms_new, rMax)

    listA, listB, d = neighborlist.neighbor_list('ijd', atoms_new, rMax)
    listMNBO_indexM=set()
    listMNBO_indexO=set()
    d_MNO = []
    for i in range(len(d)):
        if ((listA[i] < MLength) \
            and (listB[i] >= MLength)):
            d_MNO.append(d[i])
            listMNBO_indexM.add(listA[i])
            listMNBO_indexO.add(listB[i])

    dr = rMax/nBins
    edges = np.arange(0., rMax + 1.1 *dr, dr)
    h, binEdges = np.histogram(d_MNO, edges)
    #rho = len(atoms_new) / atoms.get_volume()
    #factor = 4./3. * np.pi * rho * len(atoms_new)
    rho = 0.5*(len(listMNBO_indexM)+len(listMNBO_indexO)) / atoms.get_volume()
    factor = 4./3. * np.pi * rho * 0.5*(len(listMNBO_indexM)+len(listMNBO_indexO))
    rdf = h / (factor * (binEdges[1:]**3 - binEdges[:-1]**3))

    plt.plot(binEdges[1:], rdf)
    plt.savefig('RDFADF/rdf_'+MType+'_NBO.pdf')
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
    cutoff_NBO = firstPeak*1.25
    print("first NN cutoff set to: %12.8f" % cutoff)

    #calculate CN
    NBList = neighborlist.neighbor_list('ij', atoms_new, cutoff_NBO)
    d = neighborlist.neighbor_list('d', atoms_new, cutoff_NBO)
    nnn = np.bincount(NBList[0]) #number of nearesr neighbors

    typeM_CN = np.mean(nnn[:MLength])
    typeO_CN = np.mean(nnn[MLength:])


    print("CN of %s : %8.6f" % (MType, typeM_CN))
    print("CN of %s : %8.6f" % ('NBO', typeO_CN))
    aveBondD = np.mean(d)
    stdBondD = np.std(d)
    print("averaged bond distance : %8.6f" % aveBondD)
    print("bond distance std : %8.6f" % stdBondD)
    return(cutoff, cutoff_BO, cutoff_NBO)


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
    plt.savefig('RDFADF/rdf_'+types[0]+'_'+types[1]+'.pdf')
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
    cutoff = firstPeak*1.2
    print("first NN cutoff set to: %12.8f" % cutoff)

    #calculate CN
    NBList = neighborlist.neighbor_list('ij', atoms_new, cutoff)
    nnn = np.bincount(NBList[0]) #number of nearesr neighbors
    typeA_CN = np.mean(nnn[:len(atoms_A)])
    typeB_CN = np.mean(nnn[len(atoms_A):])

    print("CN of %s : %8.6f" % (types[0], typeA_CN))
    print("CN of %s : %8.6f" % (types[1], typeB_CN))
    return(cutoff)

def get_adf_O_M_O(MType, atoms, listTypeName, listTypeNum, rMax, cutoff,\
                  dr = 1.0):
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
    MIndexList = ((NBList[0] >= MTypeStart) &\
                  (NBList[0] < MTypeEnd)).nonzero()[0]
    # O as j
    OIndexList = ((NBList[1] >= OTypeStart) &\
                  (NBList[1] < OTypeEnd)).nonzero()[0]

    BOIndexList, NBOIndexList  = get_BOList(listTypeName, listTypeNum, atoms,\
                                            cutoff)[0],\
                                 get_BOList(listTypeName, listTypeNum, atoms,\
                                            cutoff)[1]

    #print(nnn)
    #print(len(MIndexList))
    #O-M-O
    angles = []
    #print("1nd index: ")
    #print(NBList[0])
    #print("2nd index: ")
    #print(NBList[1])
    for i in range(MTypeStart, MTypeEnd):
        MOIndex = []
        if nnn[i] < 2:
            continue
        else:
            MOIndex = ((NBList[0] == i) & (NBList[1] < OTypeEnd) &\
                       (NBList[1] >= OTypeStart)).nonzero()[0]
            #print("i, MOIndex")
            #print(i, MOIndex)
            for j in range(len(MOIndex)):
                for k in range(j+1,len(MOIndex)):
                    angles.append(atoms.get_angle(NBList[1][MOIndex[j]],\
                                                  i,\
                                                  NBList[1][MOIndex[k]],\
                                                  mic=True))
                    #print(angles[-1], NBList[1][MOIndex[j]],\
                    #      i, NBList[1][MOIndex[k]])
    edges = np.arange(10.0, 180.0, dr)
    h_omo, binEdges_omo = np.histogram(angles, edges)
    plt.plot(binEdges_omo[1:], h_omo/len(angles), label = 'O-'+str(MType)+'-O')

    angles = []
    #print("1nd index: ")
    #print(NBList[0])
    #print("2nd index: ")
    #print(NBList[1])
    for i in range(MTypeStart, MTypeEnd):
        MOIndex = []
        if nnn[i] < 2:
            continue
        else:
            MOIndex = ((NBList[0] == i) & (NBList[1] < OTypeEnd) &\
                       (NBList[1] >= OTypeStart)).nonzero()[0]

            MBOIndex = np.copy(MOIndex)
            for j in range(len(MOIndex)-1,-1,-1):
                if not (NBList[1][MOIndex[j]] in BOIndexList) : 
                    MBOIndex = np.delete(MBOIndex, j)

            if (len(MBOIndex) == 0):
                continue
            else:
                #print("i, MBOIndex")
                #print(i, MBOIndex)
                for j in range(len(MBOIndex)):
                    for k in range(j+1,len(MBOIndex)):
                        angles.append(atoms.get_angle(NBList[1][MBOIndex[j]],\
                                                      i,\
                                                      NBList[1][MBOIndex[k]],\
                                                      mic=True))
                        #print(angles[-1], NBList[1][MBOIndex[j]],\
                        #      i, NBList[1][MBOIndex[k]])
    if not(len(angles) == 0):
        h_bo, binEdges_bo = np.histogram(angles, edges)
        plt.plot(binEdges_bo[1:], h_bo/len(angles), label = 'BO-'+MType+'-BO')
    else:
        print("There is no bridge-O-" + MType + \
                "-bridge-O pair for the current cutoff.")

    angles = []
    #print("1nd index: ")
    #print(NBList[0])
    #print("2nd index: ")
    #print(NBList[1])
    for i in range(MTypeStart, MTypeEnd):
        MOIndex = []
        if nnn[i] < 2:
            continue
        else:
            MOIndex = ((NBList[0] == i) & (NBList[1] < OTypeEnd) &\
                       (NBList[1] >= OTypeStart)).nonzero()[0]

            MNBOIndex = np.copy(MOIndex)
            for j in range(len(MOIndex)-1,-1,-1):
                if not (NBList[1][MOIndex[j]] in NBOIndexList) : 
                    MNBOIndex = np.delete(MNBOIndex, j)

            if (len(MNBOIndex) == 0):
                continue
            else:
                #print("i, MNBOIndex")
                #print(i, MNBOIndex)
                for j in range(len(MNBOIndex)):
                    for k in range(j+1,len(MNBOIndex)):
                        angles.append(atoms.get_angle(NBList[1][MNBOIndex[j]],\
                                      i,\
                                      NBList[1][MNBOIndex[k]],\
                                      mic=True))
                        #print(angles[-1], NBList[1][MNBOIndex[j]],\
                        #      i, NBList[1][MNBOIndex[k]])
    if not(len(angles) == 0):
        h_nbo, binEdges_nbo = np.histogram(angles, edges)
        plt.plot(binEdges_nbo[1:], h_nbo/len(angles), label = 'NBO-'+MType+'-NBO')
    else:
        print("there is no nonbridge-O-" + MType + \
                "-nonbridge-O pair for the current cutoff.")

    angles = []
    #print("1nd index: ")
    #print(NBList[0])
    #print("2nd index: ")
    #print(NBList[1])
    #print(BOIndexList)
    #print(NBOIndexList)
    for i in BOIndexList:
        MOIndex = []
        if nnn[i] < 2:
            continue
        else:
            MOIndex = ((NBList[0] == i) & (NBList[1] < OTypeEnd) &\
                       (NBList[1] >= OTypeStart)).nonzero()[0]
            MBOIndex = np.copy(MOIndex)
            MNBOIndex = np.copy(MOIndex)
            for j in range(len(MOIndex)-1,-1,-1):
                if not (NBList[1][MOIndex[j]] in BOIndexList) : 
                    MBOIndex = np.delete(MBOIndex, j)
            for j in range(len(MOIndex)-1,-1,-1):
                if not (NBList[1][MOIndex[j]] in NBOIndexList) : 
                    MNBOIndex = np.delete(MNBOIndex, j)

            if (len(MBOIndex) == 0 or len(MNBOIndex) == 0 ):
                continue
            else:
                #print("i, MBOIndex")
                #print(i, MBOIndex)
                #print("i, MNBOIndex")
                #print(i, MNBOIndex)
                for j in range(len(MBOIndex)):
                    for k in range(len(MNBOIndex)):
                        if (NBList[1][MBOIndex[j]] == NBList[1][MNBOIndex[k]]):
                            continue
                        else:
                            angles.append(atoms.get_angle(NBList[1][MBOIndex[j]],\
                                      i,\
                                      NBList[1][MNBOIndex[k]],\
                                      mic=True))
                        #print(angles[-1], NBList[1][MBOIndex[j]],\
                        #      i, NBList[1][MNBOIndex[k]])
    if not(len(angles) == 0):
        h_nbo, binEdges_nbo = np.histogram(angles, edges)
        plt.plot(binEdges_nbo[1:], h_nbo/len(angles), label = 'BO-'+MType+'-NBO')
    else:
        print("there is no nonbridge-O-" + MType + \
                "-nonbridge-O pair for the current cutoff.")


    plt.legend()
    plt.savefig('RDFADF/adf_O_'+str(MType)+'_O.pdf')
    plt.close()

    #M-O-M
    angles = []
    for i in range(OTypeStart, OTypeEnd):
        MOIndex = []
        if nnn[i] < 2:
            continue
        else:
            MOIndex = ((NBList[0] == i) & (NBList[1] < MTypeEnd) &\
                       (NBList[1] >= MTypeStart)).nonzero()[0]
            #print("i, MOIndex")
            #print(i, MOIndex)
            for j in range(len(MOIndex)):
                for k in range(j+1,len(MOIndex)):
                    angles.append(atoms.get_angle(NBList[1][MOIndex[j]],\
                                                  i,\
                                                  NBList[1][MOIndex[k]],\
                                                  mic=True))
                    #print(angles[-1], NBList[1][MOIndex[j]],\
                    #      i, NBList[1][MOIndex[k]])
    h, binEdges = np.histogram(angles, edges)

    firstMmt=getDistMoment(binEdges[1:], h/len(angles), 1, 0.0)
    secondMmt=getDistMoment(binEdges[1:], h/len(angles), 2, firstMmt)
    print("M-O-M angular distribution function:")
    print("first moment: ",firstMmt)
    print("second moment: ",secondMmt)
    print("std: ",np.sqrt(secondMmt))

    plt.plot(binEdges[1:], h/len(angles))
    plt.savefig('RDFADF/adf_'+str(MType)+'_O_'+str(MType)+'.pdf')
    plt.close()

    #M-BO-M
    angles = []
    for i in range(OTypeStart, OTypeEnd):
        MOIndex = []
        if nnn[i] < 2:
            continue
        else:
            MOIndex = ((NBList[0] == i) & (NBList[1] < MTypeEnd) &\
                       (NBList[1] >= MTypeStart)).nonzero()[0]
            for j in range(len(MOIndex)):
                for k in range(j+1,len(MOIndex)):
                    angles.append(atoms.get_angle(NBList[1][MOIndex[j]],\
                                                  i,\
                                                  NBList[1][MOIndex[k]],\
                                                  mic=True))
    h, binEdges = np.histogram(angles, edges)
    firstMmt=getDistMoment(binEdges[1:], h/len(angles), 1, 0.0)
    secondMmt=getDistMoment(binEdges[1:], h/len(angles), 2, firstMmt)
    print("M-BO-M angular distribution function:")
    print("first moment: ",firstMmt)
    print("second moment: ",secondMmt)
    print("std: ",np.sqrt(secondMmt))
    plt.plot(binEdges[1:], h/len(angles))
    plt.savefig('RDFADF/adf_'+str(MType)+'_BO_'+str(MType)+'.pdf')
    plt.close()
    with open('RDFADF/adf_'+str(MType)+'_BO_'+str(MType)+'.txt','w') as fout:
        for i in range(len(h)):
            fout.write("%12.8f %12.8f\n"%(binEdges[i+1],h[i]/len(angles)))
    return

def getDistMoment(x , y, order, eps):
    #data should have two columns, first: x(Energy) second:y(DOS)
    epsArray = [eps]*len(x)
    #print(epsArray)
    #x = np.copy(data[:,0])
    #y = np.copy(data[:,1])
    xMinusEps = np.subtract(x, epsArray)
    xToOrder = np.power(xMinusEps, order) 

    if order == 0:
        return(np.trapz(y, x))
    else:
        part1 = np.trapz(xToOrder * y, x)
        part2 = np.trapz(y, x)
        return( part1 / part2)


def get_BOList(listTypeName, listTypeNum, atoms, cutoff):
    NBList = neighborlist.neighbor_list('ijDd', atoms, cutoff)
    nnn = np.bincount(NBList[0]) #number of nearesr neighbors
    MType = 'Si'
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
    OList = np.arange(OTypeStart, OTypeEnd)
    OIndexList = ((NBList[1] >= OTypeStart) &\
                  (NBList[1] < OTypeEnd)).nonzero()[0]
    BOList = []
    for i in OList:
        if nnn[i] >=2:
            currentIndex = (NBList[0] == i).nonzero()[0]
            count = 0
            for j in (currentIndex):
                if (NBList[1][j] >= MTypeStart) and (NBList[1][j] < MTypeEnd):
                    count += 1;
            if count == 2:
                BOList.append(i)
    NBOList = list(set(OList) - set(BOList))
    return(BOList, NBOList)

def processAll(inFile, dr = 1.0):
    print("working on: %s" % inFile)
    with open(inFile,'r') as fin:
        lines=fin.readlines()
        listTypeName=[str(s) for s in lines[5].split()]
        listTypeNum=[int(s) for s in lines[6].split()]
    atoms = read(inFile)
    #get_BOList(listTypeName, listTypeNum, atoms, 2.0)
    rMax = 10.0
    cutoff = get_rdf_all(atoms, rMax) 
    try:
        cutoff = get_rdf_M_O('Si', atoms, listTypeName, listTypeNum, rMax)[0]
    except:
        cutoff = 2.0
    #cutoff = 1.65 #test only
    get_adf_O_M_O('Si', atoms, listTypeName, listTypeNum, rMax, cutoff, dr)
    #cutoff = get_rdf_M_O('Mg', atoms, listTypeName, listTypeNum, rMax)[0]
    #get_adf_O_M_O('Mg', atoms, listTypeName, listTypeNum, rMax, cutoff, dr)
    return

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("usage:\ngetRDFADF.py <vasp5 formate file name>\n")
    else:
        if (not os.path.isdir('RDFADF')):
            os.mkdir('RDFADF')
        inFile = sys.argv[1]
        #inFile = 'SiO2_c_NBO.vasp'  #test only
        processAll(inFile)

