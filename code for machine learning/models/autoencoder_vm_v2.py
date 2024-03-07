# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:32:06 2022
autoencoder_vm_v2.py

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


def model():
    encoding_dim = 500
    input_dim = 10005
    output_dim = 2001


    Input1 = Input(shape=(input_dim,))
    
    Dense1 = Dense(encoding_dim,activation='relu')(Input1)
    
   
# Variable that stores the waveform reconstructed from the encoded data
    Decoded = Dense(output_dim,activation='sigmoid')(Dense1)

# Defined as Model to reconstruct input waveform
    model = Model(inputs=[Input1], outputs=[Decoded])
    
    # model.summary()

    # visualize model
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
    model.compile(optimizer= 'Adam', loss = root_mean_squared_error, metrics = root_mean_squared_error)
    
    return model