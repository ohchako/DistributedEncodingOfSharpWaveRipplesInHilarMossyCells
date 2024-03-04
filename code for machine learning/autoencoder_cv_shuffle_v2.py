# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 13:51:53 2022

@author: Ouchi

updated; 220801 
5細胞記録なら5細胞の完全シャッフル
perform cross validation --shuffle--
version 2 for new models

"""

import numpy as np
import os

# from models.autoencoder_vm1 import model_1 # for 1 cell analysis
# from models.autoencoder_vm2 import model_2 # for 2 cells analysis
# from models.autoencoder_vm3_v2 import model_3 # for 3 cells analysis
from models.autoencoder_vm4_v2 import model_4 # for 4 cells analysis
# from models.autoencoder_vm_v2 import model    # for 5 cells analysis
 
path = 'I:\\vitro_autoencoder_data\\autoencoder_customized-main\\autoencoder_customized-main\\data4cells(in vitro)\\190927_1'
os.chdir(path)

# dataをimportする(波形ごとに[0 1]でスケーリング前処理済み)→ x
data = np.loadtxt('190927_1_4cells_normalized.txt',delimiter =',',dtype = float)
# 膜電位dataをimportする(波形ごとに[0 1]でスケーリング前処理済み)→ y
Vm = np.loadtxt('190927_1_vmtrace_normalized_4cells.txt',delimiter =',',dtype = None)
# matlabでrandperm関数で作成（インデックスが1からなので注意）
P = np.loadtxt('190927_1_Vmshuffleno.txt',delimiter =',',dtype = 'int32')

def split_list(l,i):
    for idx in range(0,l.shape[1],i):
        yield l[:,idx:idx+i]
split_vm = np.array(list(split_list(Vm,2001)))

os.chdir(path)
os.makedirs('shuffle_cv')

path_s = path + '\\shuffle_cv'
os.chdir(path_s)
os.makedirs('x_test')
os.makedirs('decoded_vm')
path_x_test = path_s + '\\x_test'
path_decoded_vm = path_s + '\\decoded_vm'

split_random = list(split_list(P,100))
p1 = split_random[0]
p2 = split_random[1]
p3 = split_random[2]
p4 = split_random[3]
# p5 = split_random[4]

###--- cross-validation
# 5回のcvですべてのSWが一回はテストデータとなるようにする
cv_times = 5
block_s = data.shape[0] // cv_times
if (block_s * cv_times) < data.shape[0]:
    block_s = block_s + 1

for x in range(10):
    path_s = path +'\\shuffle_cv'
    os.chdir(path_s)
    n = x
    P1 = p1[:,n]-1
    a2 = split_vm[0,P1]
    P2 = p2[:,n]-1
    b2 = split_vm[1,P2]
    P3 = p3[:,n]-1
    c2 = split_vm[2,P3]
    P4 = p4[:,n]-1
    d2 = split_vm[3,P4]
    # P5 = p5[:,n]-1
    # e2 = split_vm[4,P5]
    Vm_shuffle = np.concatenate([a2,b2,c2,d2],1) # 乱数表に従って順番がシャッフルされた膜電位
    
    for cv in range(cv_times):
        idx = np.arange(block_s) + ((block_s) * cv)
        if cv == int(cv_times - 1):
            idx = np.arange(block_s) + ((block_s) * cv) - (((block_s) * (cv + 1)) - data.shape[0])
            
        x_test = data[idx]
        y_test = Vm_shuffle[idx,:]
        # y_test_split = list(split_list(y_test,2001))
        # y_test_vms = np.array(y_test_split)
        
        # testに割り当てた以外のidxの9割をtraining用にする
        index_list = np.array(range(data.shape[0]))
        idx_d = np.delete(index_list,idx)
        x_train = data[idx_d]
        y_train = Vm_shuffle[idx_d,:]
        # y_train_split = list(split_list(y_train,2001))
        # y_train_vms = np.array(y_train_split)
        
        autoencoder = model_4()

        autoencoder.fit(y_train, x_train,
        epochs=100,
        batch_size=16,
        shuffle=True,
        validation_data=(y_test, x_test))
        
        # autoencoder.save(os.path.join(path_s, f"model_{cv,x}.h5".format(cv)))
        
        predicted = autoencoder.predict(y_test)
        os.chdir(path_decoded_vm)
        np.savetxt(f"decoded_vm-{cv,x}.csv",predicted,fmt='%.6f',delimiter=',')
        os.chdir(path_x_test)
        np.savetxt(f"x_test-{cv,x}.csv",x_test,fmt='%.6f',delimiter=',')
        del x_test,y_test,y_train,x_train,idx,idx_d
    del n


# from sklearn.metrics import mean_squared_error # モデル評価用(平均二乗誤差)

# rmse = []
# rmse_all = []
# for x in range(len(x_test)):
#     mse = mean_squared_error(x_test[x,:], Decoded_img[x,:]) # MSE(平均二乗誤差)の算出
#     rmse = np.sqrt(mse) # RSME = √MSEの算出
#     rmse_all = np.hstack([rmse_all,rmse])

# rmse_mean = np.mean(rmse_all)
# print('RMSE :',rmse_mean)
