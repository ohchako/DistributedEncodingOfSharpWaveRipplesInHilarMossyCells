# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:07:28 2022

update; 220802
@author: Ouchi
"""



import numpy as np

import os

import itertools

from models.autoencoder_vm1_v2 import model_1 # for 1 cell analysis
from models.autoencoder_vm2_v2 import model_2 # for 2 cells analysis
# from models.autoencoder_vm3_v2 import model_3 # for 3 cells analysis
# from models.autoencoder_vm4_v2 import model_4 # for 4 cells analysis


# modify for each data
path = 'I:\\vitro_autoencoder_data\\autoencoder_customized-main\\autoencoder_customized-main\\data3cells(in vitro)\\190927_3'
os.chdir(path)
os.makedirs('shuffle_eachcomb_cv')


# cell number
cellnum = 3
# import SW data(Scaling preprocessed by [0 1] for each waveform)→ x
data = np.loadtxt('190927_3_3cells_normalized.txt',delimiter =',',dtype = float)
# import Vm data(Scaling preprocessed by [0 1] for each waveform)→ y
Vm = np.loadtxt('190927_3_vmtrace_normalized_3cells.txt',delimiter =',',dtype = None)

index_list = np.array(range(data.shape[0]))

# random number for shuffle created with matlab's randperm function (Note that index starts from 1)
P = np.loadtxt('190927_3_Vmshuffleno.txt',delimiter =',',dtype = 'int32')

def split_list(l,i):
    for idx in range(0,l.shape[1],i):
        yield l[:,idx:idx+i]


Vm2 = np.array(list(split_list(Vm,2001)))

split_random = list(split_list(P,100))
p1 = split_random[0]
p2 = split_random[1]
# p3 = split_random[2]
# p4 = split_random[3]
# p5 = split_random[4]

###--- cross-validation
# every SW data will be used as test data within 5 cv times
cv_times = 5
block_s = data.shape[0] // cv_times
if (block_s * cv_times) < data.shape[0]:
    block_s = block_s + 1

## 10 times shuffle in each combination           
for z in range(1,cellnum):
    # if z == 2:
    #     break
    
    a = np.arange(cellnum)
    A = list(itertools.combinations(a,z))
    dirname = str(cellnum) +'C' + str(z)
    path_s = path +'\\shuffle_eachcomb_cv'
    os.chdir(path_s)
    os.makedirs(dirname)
    os.chdir(dirname)
    os.makedirs('x_test')
    os.makedirs('decoded_vm')
    path_x_test = path_s + '\\' + dirname + '\\x_test'
    path_decoded_vm = path_s + '\\' + dirname + '\\decoded_vm'
    
    for y in range(len(A)):
        p = A[y]
        Vm3 = Vm2[p,:,:]# Retrieve the necessary cells for each combination
        # Vm4 = np.hstack(Vm3) # 2dim
    
        for x in range(10):
           
            n = x
            if Vm3.shape[0] == 1:
                P1 = p1[:,n]-1
                a2 = Vm3[0,P1]
                Vm5 = a2
            elif Vm3.shape[0] == 2:
                P1 = p1[:,n]-1
                a2 = Vm3[0,P1]
                P2 = p2[:,n]-1
                b2 = Vm3[1,P2]
                Vm5 = np.concatenate([a2,b2],1)
            # elif Vm3.shape[0] == 3:
            #     P1 = p1[:,n]-1
            #     a2 = Vm3[0,P1]
            #     P2 = p2[:,n]-1
            #     b2 = Vm3[1,P2]
            #     P3 = p3[:,n]-1
            #     c2 = Vm3[2,P3]
            #     Vm5 = np.concatenate([a2,b2,c2],1)
            # elif Vm3.shape[0] == 4:
            #     P1 = p1[:,n]-1
            #     a2 = Vm3[0,P1]
            #     P2 = p2[:,n]-1
            #     b2 = Vm3[1,P2]
            #     P3 = p3[:,n]-1
            #     c2 = Vm3[2,P3]
            #     P4 = p4[:,n]-1
            #     d2 = Vm3[3,P3]
            #     Vm5 = np.concatenate([a2,b2,c2,d2],1)
            
            for cv in range(cv_times):
                idx = np.arange(block_s) + ((block_s) * cv)
                if cv == int(cv_times - 1):
                    idx = np.arange(block_s) + ((block_s) * cv) - (((block_s) * (cv + 1)) - data.shape[0])
            
                x_test = data[idx]
                y_test = Vm5[idx,:]
                # y_test_vms = np.array(list(split_list(y_test,2001)))
        
        # 90% of idx (others allocated to test) for training
                idx_d = np.delete(index_list,idx)
                x_train = data[idx_d]
                y_train = Vm5[idx_d,:]
                # y_train_vms = np.array(list(split_list(y_train,2001)))
            
                if Vm3.shape[0] == 1:
                    autoencoder = model_1()
    
                elif Vm3.shape[0] == 2:
                    autoencoder = model_2()
                
                # elif Vm3.shape[0] == 3:
                #     autoencoder = model_3()
            
                # elif Vm3.shape[0] == 4:
                #     autoencoder = model_4()
                
                autoencoder.fit(y_train, x_train,
                                epochs=100,
                                batch_size=16,
                                shuffle=True,
                                validation_data=(y_test, x_test))
                predicted = autoencoder.predict(y_test)
            
                # autoencoder.save(os.path.join(savepath, "model_{}.h5".format(cv)))
                # autoencoder.save(f"model_{z,y,x,cv}.h5".format(cv))
                os.chdir(path_decoded_vm)
                np.savetxt(f"decoded_vm-{z,y,x,cv}.csv",predicted,fmt='%.6f',delimiter=',')
                os.chdir(path_x_test)
                np.savetxt(f"x_test-{z,y,x,cv}.csv",x_test,fmt='%.6f',delimiter=',')
                del autoencoder,y_test,y_train,x_test,x_train,idx,idx_d,predicted
