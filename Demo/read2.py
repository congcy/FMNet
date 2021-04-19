import struct
import numpy as np
import random
import time #for time.sleep
import math
import scipy.misc #for imresize
import gc #for release mem
import scipy.io as sio
import sys


def norm(data):
    maxval=np.max(data)
    minval=np.min(data)
    if abs(maxval-minval) < 0.0001:
       print("wrong data\n")
       sys.exit(0);           
    return [(float(i)-minval)/float(maxval-minval) for i in data]
def norm_sgy1(data):
    return [[abs(float(i)) for i in j] for j in data]

def read_sgy(sgynam='test.sgy'):
    try:
        binsgy = open(sgynam,'r')
    except IOError:
        return 0;
    flag=1;
    return flag;
def read_file(filename='test.txt'):
    try:
        fp = open(filename,'r')
    except IOError:
        return []
    get=fp.read()
    mw = get.splitlines()
    fp.close()
    return mw;
    
def load_data(sgynam='file',sgyf1=0,sgyt1=250,step1=1,sgyf2=0,sgyt2=250,step2=1,shuffle='false'):
    data= []
    mag=[]
    ydata=[]
    for i in range(sgyf1,sgyt1+1,step1):
         filename="%s/%06d.mat" %(sgynam,i)
         print('filename = %s' %(filename))
         flag = read_sgy(filename);
         if flag != 0:
            data_mat = sio.loadmat(filename);
            data_in=data_mat['amp_3d'] #extract variables we need
            for tmp in data_in:
                tmp=np.array(tmp)
                data.append(tmp)

            mag_in=data_mat['gaus_pa'] #extract variables we need
            for tmp in mag_in:
                tmp=tmp.transpose()
                tmp=np.array(tmp)
                mag.append(tmp)	        
         else:
            print('large ev_%06d.mat not found' %(i));

    ydata=mag
    index=[i for i in range(len(ydata))]
    random.seed(7)
    if shuffle == 'true':
       random.shuffle(index)
    data = [data[i] for i in index]
    ydata = [ydata[i] for i in index]
    data=np.array(data)
    ydata=np.array(ydata)
    print("read data finished\n")
    return data.shape[1],data.shape[2],data,ydata.shape[1],ydata.shape[2],ydata

if __name__ == '__main__':
      numsgy,len1,data,ydata=load_data()
      print (numsgy,len1,len(data),data[1],data.shape,ydata.shape)
