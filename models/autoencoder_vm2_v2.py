# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:38:35 2022

autoencoder_vm2_v2.py
@author: Ouchi
"""

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from keras import backend as K
# import tensorflow_addons as tfa
# import numpy as np
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean

# import torch


def model_2():
    encoding_dim = 500
    input_dim = 4002
    output_dim = 2001


    Input1 = Input(shape=(input_dim,))
    
    Dense1 = Dense(encoding_dim,activation='relu')(Input1)
    
   
# encodeされたデータを再構成した波形を格納する変数
    Decoded = Dense(output_dim,activation='sigmoid')(Dense1)

# 入力波形を再構成するModelとして定義
    model_2 = Model(inputs=[Input1], outputs=[Decoded])
    
    # model.summary()

    # モデル可視化
    # from keras.utils import plot_model
    # plot_model(model, show_shapes = True)

    def root_mean_squared_error(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
    
    # def dtw(y_true, y_pred):
    #     y_true, y_pred = torch.FloatTensor(y_true), torch.FloatTensor(y_pred)
    #     distance,path = fastdtw(y_true,y_pred,dist=euclidean)
    #     return distance
        
    # optimizer = tfa.optimizers.AdamW(learning_rate=0.05,
                                     # weight_decay=0.001)
    model_2.compile(optimizer='adam', loss = root_mean_squared_error, metrics = root_mean_squared_error)
    
    return model_2