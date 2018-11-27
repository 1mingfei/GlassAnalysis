#!/usr/bin/python
import os,math,sys
import linecache
import numpy as np
from scipy import special
import random as rd
#import scipy.special as sp
#from matplotlib import pyplot as plt

#-------------------list of features---------------------
#nb_lst(cell,data,tot_num,i_n,Rc)
#is_float(s)
#calc_dist(lst,n1,n2,H)
#calc_dist_cart(lst,n1,n2,H)
#theta(lst,n1,n2,H) #azimuthal
#phi(lst,n1,n2,H) #polar
#Qlm_bar(lst,n1,nbl,l,m,H) #bond orientational order
#get_cos(lst,i,j,k,H)
#Class info():
#info.get_data(self)
#info.get_POSCAR(self)
#info.get_xsf(self)
#info.get_xsf_cart(self)
#info.get_surf(self) # what a junk function
#info.get_random(self) #randomly perturb atomic positions
#info.get_lmp(self,filename) #output lmp file but you need change masses
#info.get_RDF_all(self) # as you expected RDF is calculated
#info.get_RDF_list(self,lst,r_cut=6.0) #calculate RDF for atoms listed in lst
#info.get_ADF_list(self,lst,r_cut=6.0)
#info.get_ADF_all(self)
#info.get_ADF_OOO(self) #ADF for a specific type of bonds A A A
#info.get_ADF_SOS(self) #ADF for a specific type of bonds A B A
#info.get_ADF_OSO(self) #ADF for a specific type of bonds B A B
#info.get_cfg_file(self,a) #output cfg format type file
#info.get_cart(self) #get cartesian positions
#info.ellipse_aux(self,xc,yc,a,b,aux) #change cfg file aux to identify a ellipse shape
#info.circle_aux(self,xc,yc,rc,aux) #change cfg file aux to identify a circle shape
#info.rand_aux(self,conc,aux) #change cfg file aux conc should be percentile
#info.surf_aux(self,yc) #change cfg file aux for surface atoms
#info.get_BOP(self,lst,r,ll) #get bond orientational order for a listed atoms
#info.BOP_stats(self) #bop statistics
#info.get_RDF_a_b(self) #get RDF between type A and type b
#info.mv_center(self,i) # move atom i to the center of the cell
#info.wrap_back_frac(self) #in fraction positions wrap atoms back to the cell
#info.get_RDF_a_b_center(self) # totally forgot why I write this function
#---------------end list of features---------------------

def nb_lst(cell,data,tot_num,i_n,Rc):
    lst=[]
    for i in range(tot_num):
        a = calc_dist(data,i_n,i,cell) 
        if a <= Rc and (a > 0.5) and (i != i_n):
            lst.append([i,a])
    lst=np.asarray(lst)
    return lst

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
		
#define a function to calculate dist btw two atoms
def calc_dist(lst,n1,n2,H):
    if  lst[n2][1]-lst[n1][1]>=0.5:
        a=(lst[n1][1]-lst[n2][1]+1)
    elif  lst[n2][1]-lst[n1][1]<(-0.5):
        a=(lst[n1][1]-lst[n2][1]-1)
    else:
        a=((lst[n1][1]-lst[n2][1]))

    if  lst[n2][2]-lst[n1][2]>=0.5:
        b=((lst[n1][2]-lst[n2][2]+1))
    elif  lst[n2][2]-lst[n1][2]<(-0.5):
        b=((lst[n1][2]-lst[n2][2]-1))
    else:
        b=((lst[n1][2]-lst[n2][2]))

    if  lst[n2][3]-lst[n1][3]>=0.5:
        c=((lst[n1][3]-lst[n2][3]+1))
    elif  lst[n2][3]-lst[n1][3]<(-0.5):
        c=((lst[n1][3]-lst[n2][3]-1))
    else:
        c=((lst[n1][3]-lst[n2][3]))
    tmp=[a,b,c]
    tmp_cart=np.dot(tmp,H)
    aa,bb,cc=tmp_cart[0],tmp_cart[1],tmp_cart[2]

    return np.sqrt(aa**2+bb**2+cc**2)

#def calc_dist_cart(lst,n1,n2,H):
#    if  (lst[n2][0]-lst[n1][0]>=(0.5*H[0][0])):
#        a=(lst[n1][0]-lst[n2][0]+H[0][0])**2
#    elif  lst[n2][0]-lst[n1][0]<(-0.5*H[0][0]):
#        a=(lst[n1][0]-lst[n2][0]-H[0][0])**2
#    else:
#        a=(lst[n1][0]-lst[n2][0])**2
#
#    if  lst[n2][1]-lst[n1][1]>=(0.5*H[1][1]):
#        b=(lst[n1][1]-lst[n2][1]+H[1][1])**2
#    elif  lst[n2][1]-lst[n1][1]<(-0.5*H[1][1]):
#        b=(lst[n1][1]-lst[n2][1]-H[1][1])**2
#    else:
#        b=(lst[n1][1]-lst[n2][1])**2
#
#    if  lst[n2][2]-lst[n1][2]>=0.5*H[2][2]:
#        c=(lst[n1][2]-lst[n2][2]+H[2][2])**2
#    elif  lst[n2][2]-lst[n1][2]<(-0.5*H[2][2]):
#        c=(lst[n1][2]-lst[n2][2]-H[2][2])**2
#    else:
#        c=(lst[n1][2]-lst[n2][2])**2
#    return np.sqrt(a+b+c)

def calc_dist_cart(lst,n1,n2,H):
    a=(lst[n1][0]-lst[n2][0])**2
    b=(lst[n1][1]-lst[n2][1])**2
    c=(lst[n1][2]-lst[n2][2])**2
    return np.sqrt(a+b+c)


#define a function to xfer cartesian to spherial coordinate
def theta(lst,n1,n2,H): #azimuthal
    if  lst[n2][1]-lst[n1][1]>=0.5:
        a=(lst[n1][1]-lst[n2][1]+1)
    elif  lst[n2][1]-lst[n1][1]<(-0.5):
        a=(lst[n1][1]-lst[n2][1]-1)
    else:
        a=((lst[n1][1]-lst[n2][1]))
    if  lst[n2][2]-lst[n1][2]>=0.5:
        b=((lst[n1][2]-lst[n2][2]+1))
    elif  lst[n2][2]-lst[n1][2]<(-0.5):
        b=((lst[n1][2]-lst[n2][2]-1))
    else:
        b=((lst[n1][2]-lst[n2][2]))
    if  lst[n2][3]-lst[n1][3]>=0.5:
        c=((lst[n1][3]-lst[n2][3]+1))
    elif  lst[n2][3]-lst[n1][3]<(-0.5):
        c=((lst[n1][3]-lst[n2][3]-1))
    else:
        c=((lst[n1][3]-lst[n2][3]))
    tmp=[a,b,c]
    tmp_cart=np.dot(tmp,H)
    aa,bb,cc=tmp_cart[0],tmp_cart[1],tmp_cart[2]

    xy=aa**2+bb**2
    theta=np.arctan2(np.sqrt(xy),cc)
    return theta
def phi(lst,n1,n2,H): #polar
    if  lst[n2][1]-lst[n1][1]>=0.5:
        a=(lst[n1][1]-lst[n2][1]+1)
    elif  lst[n2][1]-lst[n1][1]<(-0.5):
        a=(lst[n1][1]-lst[n2][1]-1)
    else:
        a=((lst[n1][1]-lst[n2][1]))
    if  lst[n2][2]-lst[n1][2]>=0.5:
        b=((lst[n1][2]-lst[n2][2]+1))
    elif  lst[n2][2]-lst[n1][2]<(-0.5):
        b=((lst[n1][2]-lst[n2][2]-1))
    else:
        b=((lst[n1][2]-lst[n2][2]))
    if  lst[n2][3]-lst[n1][3]>=0.5:
        c=((lst[n1][3]-lst[n2][3]+1))
    elif  lst[n2][3]-lst[n1][3]<(-0.5):
        c=((lst[n1][3]-lst[n2][3]-1))
    else:
        c=((lst[n1][3]-lst[n2][3]))
    tmp=[a,b,c]
    tmp_cart=np.dot(tmp,H)
    aa,bb,cc=tmp_cart[0],tmp_cart[1],tmp_cart[2]

    phi=np.arctan2(bb,aa)
    return phi

#function to calculate bond orientational order
def Qlm_bar(lst,n1,nbl,l,m,H):   
    total = 0
    for ele in nbl:
        total+= special.sph_harm(m,l,phi(lst,n1,ele,H),theta(lst,n1,ele,H))
    return total/float(len(nbl))

def get_cos(lst,i,j,k,H):
    if  lst[j][1]-lst[i][1]>=0.5:
        a=(lst[i][1]-lst[j][1]+1)
    elif  lst[j][1]-lst[i][1]<(-0.5):
        a=(lst[i][1]-lst[j][1]-1)
    else:
        a=((lst[i][1]-lst[j][1]))
    if  lst[j][2]-lst[i][2]>=0.5:
        b=((lst[i][2]-lst[j][2]+1))
    elif  lst[j][2]-lst[i][2]<(-0.5):
        b=((lst[i][2]-lst[j][2]-1))
    else:
        b=((lst[i][2]-lst[j][2]))
    if  lst[j][3]-lst[i][3]>=0.5:
        c=((lst[i][3]-lst[j][3]+1))
    elif  lst[j][3]-lst[i][3]<(-0.5):
        c=((lst[i][3]-lst[j][3]-1))
    else:
        c=((lst[i][3]-lst[j][3]))
    tmp=[a,b,c]
    aa=np.dot(tmp,H)
    if  lst[k][1]-lst[i][1]>=0.5:
        a=(lst[i][1]-lst[k][1]+1)
    elif  lst[k][1]-lst[i][1]<(-0.5):
        a=(lst[i][1]-lst[k][1]-1)
    else:
        a=((lst[i][1]-lst[k][1]))
    if  lst[k][2]-lst[i][2]>=0.5:
        b=((lst[i][2]-lst[k][2]+1))
    elif  lst[k][2]-lst[i][2]<(-0.5):
        b=((lst[i][2]-lst[k][2]-1))
    else:
        b=((lst[i][2]-lst[k][2]))
    if  lst[k][3]-lst[i][3]>=0.5:
        c=((lst[i][3]-lst[k][3]+1))
    elif  lst[k][3]-lst[i][3]<(-0.5):
        c=((lst[i][3]-lst[k][3]-1))
    else:
        c=((lst[i][3]-lst[k][3]))
    tmp1=[a,b,c]
    bb=np.dot(tmp1,H)
    return np.dot(aa,bb)/np.linalg.norm(aa)/np.linalg.norm(bb)
    
class info(object):
    def get_data(self):
        if self.filetype == 'cfg':
            with open(self.filename,'r') as fin:
                lines= [ x for x in fin.readlines() if not x.lstrip().startswith("#") ]
            self.tot_num=int(lines[0].split()[4])
            if lines[1][0]=='A' : nl=2
            else: nl =1
            #read box dimensions
            for i in range(3):
                for j in range(3):
                    self.cell[i][j]=lines[nl+i*3+j].split()[2]
            nl+=10  # for .No Velocity.
            self.entry_count = int(lines[nl].split()[2])
            nl+=(1+self.entry_count-3)
            #get atoms coordinations
            cor = np.zeros((self.tot_num,self.entry_count+1))
            i,j=0,0
            while True:
                if (len(lines[nl+i].split())==1) and is_float(lines[nl+i].split()[0]):
                    cor[j][0]=lines[nl+i].split()[0]
                else: cor[j][0]=cor[j-1][0]
                for k in range(self.entry_count):
                    cor[j][k+1]=lines[nl+2+i].split()[k]
                i+=1
                j+=1
                if j==self.tot_num:
                    break
                if (len(lines[nl+2+i].split())==1) and is_float(lines[nl+2+i].split()[0]):
                    i+=2
            #sort atoms by type
            self.data = cor[cor[:,0].argsort()]
            del cor
            #count numbers for each type
            count=0
            atom_type_1=0
            a=self.data[0][0]
            for i in range(self.tot_num):
                if (a==self.data[i][0]) :
                    count+=1   	
                elif (a != self.data[i][0]) :
                    self.atom_type_num[atom_type_1]=count
                    a=self.data[i][0]
                    count=1
                    atom_type_1+=1
            self.atom_type_num[atom_type_1]=count

        elif self.filetype == 'xsf':
            with open ( self.filename ,'r') as fin:
                for num,line in enumerate(fin,1):
                    if 'PRIMVEC' in line:
                        for i in range(3):
                            for j in range(3):
                                self.cell[i][j]=linecache.getline(self.filename, (num+1+i)).split()[j]
        
                    if 'PRIMCOORD' in line:
                        myline = linecache.getline(self.filename, (num+1))
                        myline2 = linecache.getline(self.filename, (num+2))
                        col=int(len(myline2.split()))
                        self.tot_num=int(myline.split()[0])
                        data=[]
                        data_tmp=[]
                        for i in range(self.tot_num):
                            for j in range(col):
                                #num +2 for skipping atom numbers line
                                if j==0:
                                    data_tmp.append(str(linecache.getline(self.filename, (num+2+i)).split()[j]))
                                else:
                                    data_tmp.append(float(linecache.getline(self.filename, (num+2+i)).split()[j]))
                            self.data.append(data_tmp)
                            data_tmp=[]
            self.entry_count=int(col-4)
            #count numbers for each type
            count=0
            atom_type_1=0
            a=self.data[0][0]
            for i in range(self.tot_num):
                if (a==self.data[i][0]) :
                    count+=1   	
                elif (a != self.data[i][0]) :
                    self.atom_type_num[atom_type_1]=count
                    a=self.data[i][0]
                    count=1
                    atom_type_1+=1
            self.atom_type_num[atom_type_1]=count


        elif self.filetype == 'lmp':
            with open(self.filename,'r') as fin:
                lines= [ x for x in fin.readlines() if not x.lstrip().startswith("#") ]
            self.tot_num=int(lines[2].split()[0])
            #read box dimensions
            self.cell[0][0]=lines[6].split()[1]
            self.cell[1][1]=lines[6+1].split()[1]
            self.cell[2][2]=lines[6+2].split()[1]
            #read atoms coordinations
            cor = np.zeros((self.tot_num,4))
            for i in range(self.tot_num):
                cor[i][0]=int(lines[16+i].split()[1])
                cor[i][1]=float(lines[16+i].split()[2])/float(self.cell[0][0])
                cor[i][2]=float(lines[16+i].split()[3])/float(self.cell[1][1])
                cor[i][3]=float(lines[16+i].split()[4])/float(self.cell[2][2])
            #sort atoms by type
            self.data = cor[cor[:,0].argsort()]
            del cor
            #count numbers for each type
            count=0
            atom_type_1=0
            a=self.data[0][0]
            for i in range(self.tot_num):
                if (a==self.data[i][0]) :
                    count+=1   	
                elif (a != self.data[i][0]) :
                    self.atom_type_num[atom_type_1]=count
                    a=self.data[i][0]
                    count=1
                    atom_type_1+=1
            self.atom_type_num[atom_type_1]=count

        elif self.filetype == 'vasp' or 'POSCAR':
            with open(self.filename,'r') as fin:
                lines=fin.readlines()
            scaler=float(lines[1].split()[0])
            for i in range(3):
                for j in range(3):
                    self.cell[i][j]=scaler*float(lines[2+i].split()[j])
            lst_a=[int(s) for s in lines[6].split()]
            self.tot_num=int(sum(lst_a))
            for i in range(len(lst_a)):
                self.atom_type_num[i]=lst_a[i]
            self.data=np.zeros((self.tot_num,4))
            if lines[7][0] == 'C'  :
                for i in range(self.tot_num):
                    for j in range(3):
                        self.data[i][j+1]=float(lines[8+i].split()[j])/float(self.cell[j][j])

            elif lines[7][0] == str('D')  :
                for i in range(self.tot_num):
                    for j in range(3):
                        self.data[i][j+1]=float(lines[8+i].split()[j])

            elif lines[7][0]== 'S'  :
                if lines[8][0]== 'C' :
                    for i in range(self.tot_num):
                        for j in range(3):
                            self.data[i][j+1]=float(lines[9+i].split()[j])/float(self.cell[j][j])

                elif lines[8][0]== 'D'  :
                    for i in range(self.tot_num):
                        for j in range(3):
                            self.data[i][j+1]=float(lines[9+i].split()[j])
            num=0
            jj=0
            for i in range(len(lst_a)):
                for j in range(lst_a[i]):
                    self.data[j+jj][0]=num
                jj+=lst_a[i]
                num+=1
            self.entry_count=3
        return

    def get_POSCAR(self):
        a=str(self.filename.rsplit('.')[0])+'.vasp'
        print(a)
        with open(a,'w') as fout:
            fout.write('transfered\n')
            fout.write('1.0\n')
            for i in range(3):
                fout.write(str(str(self.cell[i][0])+' '+str(self.cell[i][1])+' '+str(self.cell[i][2])+'\n'))
            for i in range(self.atom_type):
                fout.write(str(str(int(self.atom_type_num[i]))+' '))
            fout.write('\nDirect\n')
            fout.write('\n'.join((' '.join((str(cell) for cell in x)) for x in self.data[:,[1,2,3]])))
        return
    def get_POSCAR_bot(self):
        a=str(self.filename.rsplit('.')[0])+'_SiO2.vasp'
        print(a)
        with open(a,'w') as fout:
            fout.write('transfered\n')
            fout.write('1.0\n')
            for i in range(3):
                fout.write(str(str(self.cell[i][0])+' '+str(self.cell[i][1])+' '+str(self.cell[i][2])+'\n'))
            for i in range(self.atom_type-1):
                fout.write(str(str(int(self.atom_type_num[i]))+' '))
            fout.write('\nDirect\n')
            fout.write('\n'.join((' '.join((str(cell) for cell in x)) for x in self.data[0:(int(self.atom_type_num[0])+int(self.atom_type_num[1])),[1,2,3]])))
        return

# get xsf file format
    def get_xsf(self):
        ii=str(self.filename.rsplit('.')[0])
        if os.path.isfile('energy'):
            energy_in=open('energy','r')
            lines1=energy_in.readlines()
            energy=float(line1[ii].split()[0])
            energy_in.close
        else:
            energy=0.00
        file_out=str(str(ii)+'.xsf')
        fout=open(file_out,'w')
        saaa=str('# total energy = '+ str(energy) +' eV\n\n')
        fout.write(saaa)
        fout.write('CRYSTAL\nPRIMVEC\n')
        for i in range(3):
            fout.write(str(str(self.cell[i][0])+' '+str(self.cell[i][1])+' '+str(self.cell[i][2])+'\n'))
        fout.write(str('PRIMCOORD\n'+str(self.tot_num)+' 1\n'))
        if energy==0 :
            for i in range(int(self.atom_type_num[0])):
                fout.write(str('Al  '+str(np.dot(self.data[i,1:4],self.cell[0]))+'   '+str(np.dot(self.data[i,1:4],self.cell[1]))+'   '+str(np.dot(self.data[i,1:4],self.cell[2]))+'\n'))
            for i in range(int(self.atom_type_num[0]),self.tot_num):
                fout.write(str('Ti  '+str(np.dot(self.data[i,1:4],self.cell[0]))+'   '+str(np.dot(self.data[i,1:4],self.cell[1]))+'   '+str(np.dot(self.data[i,1:4],self.cell[2]))+'\n'))
        else:
            for i in range(int(self.atom_type_num[0])):
                fout.write(str('Al  '+str(np.dot(self.data[i,1:4],self.cell[0]))+'   '+str(np.dot(self.data[i,1:4],self.cell[1]))+'   '+str(np.dot(self.data[i,1:4],self.cell[2]))+'   '+str(self.data[i][4])+'   '+str(self.data[i][5])+'   '+str(self.data[i][6])+'\n'))
            for i in range(int(self.atom_type_num[0]),self.tot_num):
                fout.write(str('Ti  '+str(np.dot(self.data[i,1:4],self.cell[0]))+'   '+str(np.dot(self.data[i,1:4],self.cell[1]))+'   '+str(np.dot(self.data[i,1:4],self.cell[2]))+'   '+str(self.data[i][4])+'   '+str(self.data[i][5])+'   '+str(self.data[i][6])+'\n'))
        return

    def get_xsf_cart(self):
        ii=str(self.filename.rsplit('.')[0])
        if os.path.isfile('energy'):
            energy_in=open('energy','r')
            lines1=energy_in.readlines()
            energy=float(line1[ii].split()[0])
            energy_in.close
        else:
            energy=0.00
        file_out=str(str(ii)+'.xsf')
        fout=open(file_out,'w')
        saaa=str('# total energy = '+ str(energy) +' eV\n\n')
        fout.write(saaa)
        fout.write('CRYSTAL\nPRIMVEC\n')
        for i in range(3):
            fout.write(str(str(self.cell[i][0])+' '+str(self.cell[i][1])+' '+str(self.cell[i][2])+'\n'))
        fout.write(str('PRIMCOORD\n'+str(self.tot_num)+' 1\n'))
        if energy==0 :
            for i in range(int(self.atom_type_num[0])):
                fout.write('Al  %12.6f  %12.6f  %12.6f\n' %(self.data_cart[i][0],self.data_cart[i][1],self.data_cart[i][2]))
            for i in range(int(self.atom_type_num[0]),self.tot_num):
                fout.write('Ti  %12.6f  %12.6f  %12.6f\n' %(self.data_cart[i][0],self.data_cart[i][1],self.data_cart[i][2]))
        else:
            for i in range(int(self.atom_type_num[0])):
                fout.write('Al  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f\n' %(self.data_cart[i][0],self.data_cart[i][1],self.data_cart[i][2],self.data[i][4],self.data[i][2],self.data[i][6]))
            for i in range(int(self.atom_type_num[0]),self.tot_num):
                fout.write('Ti  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f\n' %(self.data_cart[i][0],self.data_cart[i][1],self.data_cart[i][2],self.data[i][4],self.data[i][2],self.data[i][6]))
        return

    def get_surf(self):
        new_cell_z=self.cell[2][2]+12.0
        for i in range(self.tot_num):
            tmp=np.dot(self.data[i,1:4],self.cell[2])
            tmp+=6.0
            tmp/=float(new_cell_z)
            self.data[i][3]=tmp
            
        self.cell[2][2]+=12.0
        return

    def get_random(self):
        a=rd.random()
        for i in range(self.tot_num):
            self.data[i][3]+=a
            if self.data[i][3]> 1.0: self.data[i][3]-=1.0
        return

    def get_lmp(self,filename):
        with open(filename,'w') as fout:
            fout.write('comments\n\n    '+str(self.tot_num)+'    atoms\n\n   2   atom types\n\n')
            fout.write('0.000   '+str(self.cell[0][0])+'  xlo xhi\n')
            fout.write('0.000   '+str(self.cell[1][1])+'  ylo yhi\n')
            fout.write('0.000   '+str(self.cell[2][2])+'  zlo zhi\n'+str("%12.6f"% self.cell[1][0])+'  '+str("%12.6f" % self.cell[2][0])+'  '+str("%12.6f" % self.cell[1][2])+'   xy xz yz\n\n    Masses\n\n   1  15.999400\n   2   28.085501\n\nAtoms\n\n')
            nm=1
            for i in range(int(self.atom_type_num[0])):
                fout.write('    '+str(nm)+'     1       '+str(np.dot(self.data[i,1:4],self.cell[0]))+'    '+str(np.dot(self.data[i,1:4],self.cell[1]))+'    '+str(np.dot(self.data[i,1:4],self.cell[2]))+'\n')
                nm+=1
            for i in range(int(self.atom_type_num[0]),self.tot_num):
                fout.write('    '+str(nm)+'     2       '+str(np.dot(self.data[i,1:4],self.cell[0]))+'    '+str(np.dot(self.data[i,1:4],self.cell[1]))+'    '+str(np.dot(self.data[i,1:4],self.cell[2]))+'\n')
                nm+=1
        return

    def get_RDF_all(self):
        dr=0.02
        r_cut=6.0
        rho=1.0
        num={}
        nb=int(r_cut/dr)
        for i in range(nb):
            num[str(i)]=0
        for i in range(self.tot_num-1):
            for j in range(i+1,self.tot_num):
                if self.filetype == 'xsf':
                    a= calc_dist_cart(self.data,i,j,self.cell)
                else:
                    a= calc_dist(self.data,i,j,self.cell)
                if   a < r_cut:
                    binn=int(a/dr)
                    num[str(binn)]+=2
        for i in range(nb):
            b=4/3.0*np.pi*((i+1)**3-i**3)*dr**3*rho
#            b=4*np.pi*(i+1)*dr**2*(i+2)*dr*rho
            print(i*dr,float(num[str(i)])/float(self.tot_num)/b)
        return

    def get_RDF_list(self,lst,r_cut=6.0):
        dr=0.02
        rho=1.0
        num={}
        nb=int(r_cut/dr)
        for i in range(nb):
            num[str(i)]=0
        for i in lst:
            for j in range(i+1,self.tot_num):
                if self.filetype == 'xsf':
                    a= calc_dist_cart(self.data,i,j,self.cell)
                else:
                    a= calc_dist(self.data,i,j,self.cell)
                if   a < r_cut:
                    binn=int(a/dr)
                    num[str(binn)]+=2
        with open('rdf_all.dat','w')as fin:
            for i in range(nb):
                b=4/3.0*np.pi*((i+1)**3-i**3)*dr**3*rho
                fin.write("%12.8f  %12.8f\n"%(i*dr,float(num[str(i)])/float(self.tot_num)/b))     
        return


    def get_nb_lst_dist(self,lst,r_cut):
        N=len(lst)
        nb_l_d=np.zeros((N,1+20+20))
        for i in lst:
            i_lst=nb_lst(self.cell,self.data,self.tot_num,i,r_cut)#[:,0] neighbor id [:,1]neighbor distance
            nb_l_d[i][0]=len(i_lst)
            for j in range(len(i_lst)):
                nb_l_d[i][1+j]=i_lst[j][0]
                nb_l_d[i][21+j]=i_lst[j][1]
        return(nb_l_d)

    def get_ADF_list(self,lst,r_cut=6.0):
        # interval dr of 1 degree
        dr=1.0 # degree
        num={}
        for i in range(0,180,1):
            num[str(i)]=0
        #for i in range(self.tot_num-1):
        for i in lst:
            i_lst=nb_lst(self.cell,self.data,self.tot_num,i,r_cut)[:,0]
            print(len(i_lst))
            print(i_lst)
            for j in i_lst:
                j=int(j)
                for k in i_lst:
                    k=int(k)
                    if k!=i and k!= j:
                        if calc_dist(self.data,k,j,self.cell) < r_cut and calc_dist(self.data,k,i,self.cell) < r_cut :
                            tmp=(int(np.arccos(get_cos(self.data,k,j,i,self.cell))/np.pi*180))
                            if tmp==180:tmp=0
                            num[str(tmp)]+=1
        with open('adf_all.dat','w')as fin:
            for i in range(0,180,1):
                fin.write("%12.8f  %12.8f\n"%(float(i),float(num[str(i)])))
        return

    def get_ADF_all(self):
        # interval dr of 1 degree
        dr=1.0 # degree
        r_cut=3.0
        num={}
        for i in range(0,180,1):
            num[str(i)]=0
        for i in range(self.tot_num-1):
            for j in range(i+1,self.tot_num):
                    for k in range(self.tot_num):
                        if k!=i and k!= j:
                            if calc_dist(self.data,k,j,self.cell) < r_cut and calc_dist(self.data,k,i,self.cell) < r_cut :
                                #num[str(int(np.arccos(get_cos(self.data,i,j,k,self.cell))/np.pi*180))]+=1
                                #num[str(int(np.arccos(get_cos(self.data,j,i,k,self.cell))/np.pi*180))]+=1
                                num[str(int(np.arccos(get_cos(self.data,k,j,i,self.cell))/np.pi*180))]+=1
        for i in range(0,180,1):
            print(i,num[str(i)])
        return

    def get_ADF_OOO(self):
        # interval dr of 1 degree
        dr=1.0 # degree
        r_cut=3.0
        num={}
        for i in range(0,180,1):
            num[str(i)]=0
        for i in range(64):
            for j in range(64):
                    for k in range(64):
                        if k!=i and k!= j and i!=j:
                            if calc_dist(self.data,k,j,self.cell) < r_cut and calc_dist(self.data,k,i,self.cell) < r_cut :
                                #num[str(int(np.arccos(get_cos(self.data,i,j,k,self.cell))/np.pi*180))]+=1
                                #num[str(int(np.arccos(get_cos(self.data,j,i,k,self.cell))/np.pi*180))]+=1
                                num[str(int(np.arccos(get_cos(self.data,k,j,i,self.cell))/np.pi*180))]+=1
        for i in range(0,180,1):
            print(i,num[str(i)])
        return

    def get_ADF_SOS(self):
        # interval dr of 1 degree
        dr=1.0 # degree
        r_cut=3.0
        num={}
        for i in range(0,180,1):
            num[str(i)]=0
        for i in range(64,96):
            for j in range(64,96):
                    for k in range(64):
                        if k!=i and k!= j and i!=j:
                            if calc_dist(self.data,k,j,self.cell) < r_cut and calc_dist(self.data,k,i,self.cell) < r_cut :
                                #num[str(int(np.arccos(get_cos(self.data,i,j,k,self.cell))/np.pi*180))]+=1
                                #num[str(int(np.arccos(get_cos(self.data,j,i,k,self.cell))/np.pi*180))]+=1
                                num[str(int(np.arccos(get_cos(self.data,k,j,i,self.cell))/np.pi*180))]+=1
        for i in range(0,180,1):
            print(i,num[str(i)])
        return

    def get_ADF_OSO(self):
        # interval dr of 1 degree
        dr=1.0 # degree
        r_cut=3.0
        num={}
        for i in range(0,180,1):
            num[str(i)]=0
        for i in range(64):
            for j in range(64):
                    for k in range(64,96):
                        if k!=i and k!= j and i!=j:
                            if calc_dist(self.data,k,j,self.cell) < r_cut and calc_dist(self.data,k,i,self.cell) < r_cut :
                                #num[str(int(np.arccos(get_cos(self.data,i,j,k,self.cell))/np.pi*180))]+=1
                                #num[str(int(np.arccos(get_cos(self.data,j,i,k,self.cell))/np.pi*180))]+=1
                                num[str(int(np.arccos(get_cos(self.data,k,j,i,self.cell))/np.pi*180))]+=1
        for i in range(0,180,1):
            print(i,num[str(i)])
        return

    def get_cfg_file(self,a):
        #print(self.data)
        #print(self.data.shape)
        with open(a,'w') as fout:
            fout.write(str('Number of particles = '+str(self.tot_num)+'\n'))
            fout.write(str('H0(1,1) = '+str(self.cell[0][0])+' A\n'))
            fout.write(str('H0(1,2) = '+str(self.cell[0][1])+' A\n'))
            fout.write(str('H0(1,3) = '+str(self.cell[0][2])+' A\n'))
            fout.write(str('H0(2,1) = '+str(self.cell[1][0])+' A\n'))
            fout.write(str('H0(2,2) = '+str(self.cell[1][1])+' A\n'))
            fout.write(str('H0(2,3) = '+str(self.cell[1][2])+' A\n'))
            fout.write(str('H0(3,1) = '+str(self.cell[2][0])+' A\n'))
            fout.write(str('H0(3,2) = '+str(self.cell[2][1])+' A\n'))
            fout.write(str('H0(3,3) = '+str(self.cell[2][2])+' A\n'))
            fout.write('.NO_VELOCITY.\n')
            fout.write('entry_count = %i\n' % int(self.entry_count))
            #fout.write('auxiliary[0] =  fx [eV/A]\n')
            #fout.write('auxiliary[1] =  fy [eV/A]\n')
            #fout.write('auxiliary[2] =  fz [eV/A]\n')
            for i in range(self.entry_count-3):
                fout.write('auxiliary[%i] =  %s  \n' % (int(i), str((i))))
            b=self.data[0][0]
            fout.write(str(str(b)+'\nW\n')) 
            for i in range(self.tot_num):
                if (self.data[i][0] != b): 
                    fout.write(str(str(self.data[i][0])+'\nAg\n'))
                for j in range(1,1+self.entry_count):
                    #fout.write(str(str(self.data[i][j])+' '))
                    fout.write('%+18.10E  '%(self.data[i][j]))
                fout.write('\n')
                b=self.data[i][0]
        return

    def get_cart(self):
        self.pos_cart=np.dot(self.data[:,1:4],self.cell)
        return self.pos_cart

    def ellipse_aux(self,xc,yc,a,b,aux):
        for i in range(self.tot_num):
            if self.data[i][0] != self.data[0][0]:
                if ((self.data[i][1]-xc)**2.0) / (a**2.0) + \
                        ((self.data[i][2]-yc)**2.0) / (b**2.0)<=1.0:
                    self.data[i][aux]=0.9
        return

    def circle_aux(self,xc,yc,rc,aux):
        for i in range(self.tot_num):
            if self.data[i][0] != self.data[0][0]:
                if (self.pos_cart[i][0]-xc*self.cell[0][0])**2.0 + \
                        (self.pos_cart[i][1]-yc*self.cell[1][1])**2.0<=rc**2.0:
                    self.data[i][aux]=0.9
        return

    def rand_aux(self,conc,aux): # conc should be percentile
        typ=1
        tot=0.0
        for i in range(int(self.atom_type_num[1])):
            if self.data[int(self.atom_type_num[0])+i][aux] != 0.9 and\
                    rd.random() < conc:
                self.data[int(self.atom_type_num[0])+i][aux] = 0.9
            tot+=self.data[int(self.atom_type_num[0])+i][aux]
        print(tot/self.atom_type_num[1])
        return

    def surf_aux(self,yc):
        for i in range(self.tot_num):
            if (self.data[i][3])>=yc:
                self.data[i][4]=0.1
                self.data[i][5]=0.5
        return


    def get_BOP(self,lst,r,ll):
        #calculate BOP from Q2 ,Q4, Q6 to Qll
        a=str(self.filename.rsplit('.')[0]+'_bop.cfg')
        print(a)
        Qn=np.zeros((int(len(lst)),int(ll/2)))
        for l in range(2,ll+1,2):
            nb_dict={}
            Q=0
            for j in lst:
                total = 0
                nb_dict[str(j)]=[]
                for i in range(self.tot_num):
                    if (calc_dist(self.data,j,i,self.cell)<= r)and(j!=i):
                        nb_dict[str(j)].append(i)
                for i in range(-l,l+1,1):
                    total += Qlm_bar(self.data, j, nb_dict[str(j)], l, i,\
                            self.cell) * \
                            np.conj(Qlm_bar(self.data, j, nb_dict[str(j)],\
                            l,i,self.cell))
                total*= ((4*np.pi)/(2*l+1))
                Qn[j][int(l/2-1)]= np.real(np.sqrt(total))
                #print(len(nb_dict[str(j)]))
        self.data=np.append(self.data[:len(lst)],Qn,axis=1)
        with open(a,'w') as fout:
            fout.write(str('Number of particles = '+str(len(lst))+'\n'))
            fout.write(str('H0(1,1) = '+str(self.cell[0][0])+' A\n'))
            fout.write(str('H0(1,2) = '+str(self.cell[0][1])+' A\n'))
            fout.write(str('H0(1,3) = '+str(self.cell[0][2])+' A\n'))
            fout.write(str('H0(2,1) = '+str(self.cell[1][0])+' A\n'))
            fout.write(str('H0(2,2) = '+str(self.cell[1][1])+' A\n'))
            fout.write(str('H0(2,3) = '+str(self.cell[1][2])+' A\n'))
            fout.write(str('H0(3,1) = '+str(self.cell[2][0])+' A\n'))
            fout.write(str('H0(3,2) = '+str(self.cell[2][1])+' A\n'))
            fout.write(str('H0(3,3) = '+str(self.cell[2][2])+' A\n'))
            fout.write('.NO_VELOCITY.\n')
            fout.write('entry_count = %i\n' % int(self.entry_count+ll/2))
            #fout.write('auxiliary[0] =  fx [eV/A]\n')
            #fout.write('auxiliary[1] =  fy [eV/A]\n')
            #fout.write('auxiliary[2] =  fz [eV/A]\n')
            #for i in range(ll/2):
            #    fout.write('auxiliary[%i] =  %s  \n' % (int(3+i), str(2*(i+1))))
            for i in range(int(ll/2)):
                fout.write('auxiliary[%i] =  %s  \n' % (int(i), str(2*(i+1))))
            b=self.data[0][0]
            fout.write(str(str(b)+'\nW\n')) 
            for i in range(len(lst)):
                if (self.data[i][0] != b):
                    fout.write(str(str(self.data[i][0])+'\nSi\n'))
                for j in range(1,1+int(ll/2)+self.entry_count):
                    fout.write(str(str(self.data[i][j])+' '))
                fout.write('\n')
                b=self.data[i][0]
        Qn=np.asarray(Qn)
        return(Qn)

    def BOP_stats(self):
        st=np.zeros((int(self.cell[2][2]),(self.entry_count-2)))
        for i in range(int(self.cell[2][2])):
            tot_pool=np.zeros(self.entry_count-3)
            num_pool=np.zeros(self.entry_count-3)
            ave_pool=np.zeros(self.entry_count-3)
            for j in range(self.tot_num):
                if self.data[j][3]*self.cell[2][2] <= float(i+1) and \
                        self.data[j][3]*self.cell[2][2] > float(i):
                    for k in range(self.entry_count-3):
                        tot_pool[k]+=self.data[j][3+k]
                        num_pool[k]+=1
            for k in range(self.entry_count-3):
                if num_pool[k]==0:
                    ave_pool[k]=0
                else:
                    ave_pool[k]=float(tot_pool[k])/float(num_pool[k])
                    
            st[i][0]=i
            for k in range(self.entry_count-3):
                st[i][k+1]=ave_pool[k]
        print(st)
        for k in range(self.entry_count-3):
            str1=str(2*(k+1))
            plt.plot(st[:,0],st[:,(k+1)],label=str1)

        plt.suptitle('All', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.savefig('depth_BOP.png')
        plt.close()

        st=np.zeros((int(self.cell[2][2]/2),(self.entry_count-2)))
        for i in range(int(self.cell[2][2]/2)):
            tot_pool=np.zeros(self.entry_count-3)
            num_pool=np.zeros(self.entry_count-3)
            ave_pool=np.zeros(self.entry_count-3)
            for j in range(64):
                if self.data[j][3]*self.cell[2][2] <= float(2*i+2) and  self.data[j][3]*self.cell[2][2] > float(2*i):
                    for k in range(self.entry_count-3):
                        tot_pool[k]+=self.data[j][3+k]
                        num_pool[k]+=1
            for k in range(self.entry_count-3):
                if num_pool[k]==0:
                    ave_pool[k]=0
                else:
                    ave_pool[k]=float(tot_pool[k])/float(num_pool[k])
                    
            st[i][0]=2*i
            for k in range(self.entry_count-3):
                st[i][k+1]=ave_pool[k]
        print(st)
        for k in range(self.entry_count-3):
            str1=str(2*(k+1))
            plt.plot(st[:,0],st[:,(k+1)],label=str1)

        plt.suptitle('O', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.savefig('O_depth_BOP.png')
        plt.close()

        st=np.zeros((int(self.cell[2][2]/2),(self.entry_count-2)))
        for i in range(int(self.cell[2][2]/2)):
            tot_pool=np.zeros(self.entry_count-3)
            num_pool=np.zeros(self.entry_count-3)
            ave_pool=np.zeros(self.entry_count-3)
            for j in range(64,96):
                if self.data[j][3]*self.cell[2][2] <= float(2*i+2) and  self.data[j][3]*self.cell[2][2] > float(2*i):
                    for k in range(self.entry_count-3):
                        tot_pool[k]+=self.data[j][3+k]
                        num_pool[k]+=1
            for k in range(self.entry_count-3):
                if num_pool[k]==0:
                    ave_pool[k]=0
                else:
                    ave_pool[k]=float(tot_pool[k])/float(num_pool[k])
                    
            st[i][0]=2*i
            for k in range(self.entry_count-3):
                st[i][k+1]=ave_pool[k]
        print(st)
        for k in range(self.entry_count-3):
            str1=str(2*(k+1))
            plt.plot(st[:,0],st[:,(k+1)],label=str1)
        plt.suptitle('Si', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.savefig('Si_depth_BOP.png')
        plt.close()
        return

    def get_RDF_a_b(self):
        dr=0.02
        r_cut=6.0
        rho=1.0
        num={}
        nb=int(r_cut/dr)
        for i in range(nb):
            num[str(i)]=0
#        print(self.atom_type_num[0])
#        print(self.atom_type_num[1])
        for i in range(int(self.atom_type_num[0])):
            for j in range(int(self.atom_type_num[0]),self.tot_num):
                if self.filetype == 'xsf':
                    a= calc_dist_cart(self.data_cart,i,j,self.cell)
                else:
                    a= calc_dist(self.data,i,j,self.cell)
                if   a < r_cut:
                    binn=int(a/dr)
                    num[str(binn)]+=2
        nm1='rdf_Al_Ti_'+str(self.pass_val)
        f=open(nm1,'w')
        for i in range(nb):
            b=4/3.0*np.pi*((i+1)**3-i**3)*dr**3*rho
#            b=4*np.pi*(i+1)*dr**2*(i+2)*dr*rho
            f.write('%12.8f     %12.8f\n'%(i*dr,float(num[str(i)])/float(self.tot_num)/b))
            #print(i*dr,float(num[str(i)])/float(self.tot_num)/b)
        f.close()
        return

    def mv_center(self,i):
        center=[0.5,0.5,0.5]
        center_cart=np.dot(center,self.cell)
        dist=[self.data[i][1]-center_cart[0],
                self.data[i][2]-center_cart[1],
                self.data[i][3]-center_cart[2]]
        for j in range(self.tot_num):
            self.data[j][1]-=dist[0]
            self.data[j][2]-=dist[1]
            self.data[j][3]-=dist[2]
        data_tmp=np.zeros((self.tot_num,3))
        for i in range(self.tot_num):
            data_tmp[i][0]= self.data[i][1]
            data_tmp[i][1]= self.data[i][2]
            data_tmp[i][2]= self.data[i][3]
        self.data_frac=np.dot(data_tmp,np.linalg.inv(self.cell))
        print(self.data_frac)
        self.wrap_back_frac()
        self.data_cart=np.dot( self.data_frac, self.cell)
        return
 
    def wrap_back_frac(self):
        for i in range(self.tot_num):
            a=math.floor(self.data_frac[i][0])
            if a > 1e-10 :
                self.data_frac[i][0] -= a
            b=math.floor(self.data_frac[i][1])
            if b > 1e-10 :
                self.data_frac[i][1] -= b
            c=math.floor(self.data_frac[i][2])
            if c > 1e-10 :
                self.data_frac[i][2] -= c
        return


    def get_RDF_a_b_center(self):
        dr=0.02
        r_cut=6.0
        rho=1.0
        num={}
        nb=int(r_cut/dr)
        for i in range(nb):
            num[str(i)]=0
        for i in range(int(self.atom_type_num[0])):
            #if ((self.data[i][3]>44.0) and (self.data[i][3]<64.0)):
            if ((self.data[i][3]>51.0) and (self.data[i][3]<59.0)):
                self.mv_center(i)
                self.wrap_back_frac()
                for j in range(int(self.atom_type_num[0]),int(self.tot_num)):
                    if self.filetype == 'xsf':
                        a= calc_dist_cart(self.data_cart,i,j,self.cell)
                    else:
                        a= calc_dist(self.data,i,j,self.cell)
                    if   a < r_cut:
                        binn=int(a/dr)
                        num[str(binn)]+=2
        nm1='rdf_Al_Ti_center_'+str(self.pass_val)
        f=open(nm1,'w')
        for i in range(nb):
            b=4/3.0*np.pi*((i+1)**3-i**3)*dr**3*rho
#            b=4*np.pi*(i+1)*dr**2*(i+2)*dr*rho
            f.write('%12.8f     %12.8f\n'%(i*dr,float(num[str(i)])/float(self.tot_num)/b))
            #print(i*dr,float(num[str(i)])/float(self.tot_num)/b)
        f.close()
        return
   
    def __init__(self,filename,filetype,atom_type,pass_val=0):
        self.filename=str(filename)
        self.filetype=filetype
        self.tot_num=0
        self.atom_type=atom_type
        self.pass_val=pass_val
        self.cell=np.zeros((3,3))
        self.data=[]
        self.data_cart=[]
        self.data_frac=[]
        self.entry_count=0  # how many attributes beside atom type and coordinates
        self.atom_type_num=np.zeros(atom_type)
        self.get_data()
        return


#if __name__ == "__main__":
#    aa=str(raw_input('filename, # of atomtype  and output format  please:\n'))
#    if (len(aa.split())==3):
#        sa=aa.split()[0]
#        typ_in=str(sa.split(".")[1])
#        na=int(aa.split()[1])
#        ta=aa.split()[2]
#        a=info(sa,typ_in,na)
#        if ta=='vasp':
#            a.get_POSCAR()
#        if ta=='xsf':
#            a.get_xsf()
#    elif (len(aa.split())==2):
#        sa=aa.split()[0]
#        na=int(aa.split()[1])
#        a=info(sa,'cfg',na)
#        a.get_RDF_all()
#        a.get_BOP(4,4)
