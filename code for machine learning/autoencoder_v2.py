# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 19:29:44 2022

@author: Ouchi

basic code for autoencoder
"""



import numpy as np
import os

# import itertools
# import matplotlib.pyplot as plt # for confirmation

# from models.autoencoder_vm1 import model_1 # for 1 cell analysis
# from models.autoencoder_vm2 import model_2 # for 2 cells analysis
from models.autoencoder_vm3_v2 import model_3 # for 3 cells analysis
# from models.autoencoder_vm4_v2 import model_4 # for 4 cells analysis
# from models.autoencoder_vm_v2 import model    # for 5 cells analysis


# modify for each data
path = 'I:\\vitro_autoencoder_data\\autoencoder_customized-main\\autoencoder_customized-main\\data3cells(in vitro)\\190927_3'



# import SW data(Scaling preprocessed by [0 1] for each waveform)→ x
data = np.loadtxt('190927_3_3cells_normalized.txt',delimiter =',',dtype = float)

# for confirmation
# plt.figure()
# plt.plot(Vm[1,:], color = "black")

# import Vm data(Scaling preprocessed by [0 1] for each waveform)→ y
Vm = np.loadtxt('190927_3_vmtrace_normalized_3cells.txt',delimiter =',',dtype = None)

os.chdir(path)
os.makedirs('D500')
path_s = path + '\\D500'
os.chdir(path_s)
os.makedirs('x_test')
os.makedirs('decoded_vm')
path_x_test = path_s + '\\x_test'
path_decoded_vm = path_s + '\\decoded_vm'


###--- cross-validation
# every SW data will be used as test data within 5 cv times
cv_times = 5
block_s = data.shape[0] // cv_times
if (block_s * cv_times) < data.shape[0]:
    block_s = block_s + 1

            
for cv in range(cv_times):
    idx = np.arange(block_s) + ((block_s) * cv)
    if cv == int(cv_times - 1):
        idx = np.arange(block_s) + ((block_s) * cv) - (((block_s) * (cv + 1)) - data.shape[0])
           
    x_test = data[idx]
    y_test_vms = Vm[idx,:]
        
        # 90% of idx (others allocated to test) for training
    index_list = np.array(range(data.shape[0]))
    idx_d = np.delete(index_list,idx)
    x_train = data[idx_d]
    y_train_vms = Vm[idx_d,:]
                
    autoencoder = model_3()
                
    autoencoder.fit(y_train_vms, x_train,
                    epochs=100,
                    batch_size=16,
                    shuffle=True,
                    validation_data=(y_test_vms, x_test))
    predicted = autoencoder.predict(y_test_vms)
    
            
    os.chdir(path_decoded_vm)
    np.savetxt(f"D500_cv-{cv}.csv",predicted,fmt='%.6f',delimiter=',')
    os.chdir(path_x_test)
    np.savetxt(f"D500_cv_x_test-{cv}.csv",x_test,fmt='%.6f',delimiter=',')
    del autoencoder,x_test,idx,idx_d,y_test_vms,y_train_vms
