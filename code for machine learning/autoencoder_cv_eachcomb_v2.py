# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:45:50 2022

autoencoder_cv_eachcomb_v2.py
@author: Ouchi
"""


import numpy as np

# from sklearn import preprocessing
#import matplotlib.pyplot as plt
#import time
# from pylab import rcParams

import os

import itertools

from models.autoencoder_vm1_v2 import model_1
from models.autoencoder_vm2_v2 import model_2
from models.autoencoder_vm3_v2 import model_3
from models.autoencoder_vm4_v2 import model_4

# modify for each data  
path = 'I:\\vitro_autoencoder_data\\autoencoder_customized-main\\autoencoder_customized-main\\data3cells(in vitro)\\190924_2'
os.chdir(path)
os.makedirs('combination_cv')

# import SW data(Scaling preprocessed by [0 1] for each waveform)→ x
data = np.loadtxt('190924_2_3cells_normalized.txt',delimiter =',',dtype = float)
# import Vm data(Scaling preprocessed by [0 1] for each waveform)→ y
Vm = np.loadtxt('190924_2_vmtrace_normalized_3cells.txt',delimiter =',',dtype = None)

index_list = np.array(range(data.shape[0]))

def split_list(l,i):
    for idx in range(0,l.shape[1],i):
        yield l[:,idx:idx+i]


# cell number
cellnum = 3
Vm2 = np.array(list(split_list(Vm,2001)))

###--- cross-validation
# every SW data will be used as test data within 5 cv times
cv_times = 5
block_s = data.shape[0] // cv_times
if (block_s * cv_times) < data.shape[0]:
    block_s = block_s + 1
          
for z in range(1,cellnum): # create file for each combination
    
    a = np.arange(cellnum)
    A = list(itertools.combinations(a,z))
    dirname = str(cellnum) +'C' + str(z)
    path_s = path +'\\combination_cv'
    os.chdir(path_s)
    os.makedirs(dirname)
    os.chdir(dirname)
    os.makedirs('x_test')
    os.makedirs('decoded_vm')
    path_x_test = path_s + '\\' + dirname + '\\x_test'
    path_decoded_vm = path_s + '\\' + dirname + '\\decoded_vm'
    
    for y in range(len(A)):
        p = A[y]
        Vm3 = Vm2[p,:,:]# retrieve the necessary cells for each combination
        Vm4 = np.hstack(Vm3) # 2dim
        
        for cv in range(cv_times):
            idx = np.arange(block_s) + ((block_s) * cv)
            if cv == int(cv_times - 1):
                idx = np.arange(block_s) + ((block_s) * cv) - (((block_s) * (cv + 1)) - data.shape[0])

            x_test = data[idx]
            y_test = Vm4[idx,:]
            # y_test_vms = np.array(list(split_list(y_test,2001)))
            
        # 90% of idx (others allocated to test) for training
            idx_d = np.delete(index_list,idx)
            x_train = data[idx_d]
            y_train = Vm4[idx_d,:]
            # y_train_vms = np.array(list(split_list(y_train,2001)))
       
            if Vm3.shape[0] == 1:
                autoencoder = model_1()
            
            elif Vm3.shape[0] == 2:
                autoencoder = model_2()
                
            elif Vm3.shape[0] == 3:
                autoencoder = model_3()
            
            elif Vm3.shape[0] == 4:
                autoencoder = model_4()
                
            autoencoder.fit(y_train, x_train,
                            epochs=100,
                            batch_size=16,
                            shuffle=True,
                            validation_data=(y_test, x_test))
            predicted = autoencoder.predict(y_test)
            
            # autoencoder.save(os.path.join(dirname, f"model_{z,y,cv}.h5".format(cv)))
            # autoencoder.save(f"model_{z,y,cv}.h5".format(cv))
            os.chdir(path_decoded_vm)
            np.savetxt(f"combination_cv-{z,y,cv}.csv",predicted,fmt='%.6f',delimiter=',')
            os.chdir(path_x_test)
            np.savetxt(f"combination_cv_x_test-{z,y,cv}.csv",x_test,fmt='%.6f',delimiter=',')
            del autoencoder,y_test,y_train,x_test,idx,idx_d
