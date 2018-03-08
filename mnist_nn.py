# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 09:31:38 2018

@author: schiejak
"""

#Importing libraries
import tflearn 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import numpy as np
import tensorflow as tf




#Preparing training data to feed into neural network
feed = np.load('train_data.npy')    
X = np.array([i[0] for i in feed]).reshape(-1, 28, 28, 1)
Y = [i[1] for i in feed]    
    
#Neural network structure

tf.reset_default_graph()


sit = input_data(shape=[None, 28, 28, 1])

sit = conv_2d(sit, 56, 3, activation='relu') 
sit = conv_2d(sit, 56, 3, activation='relu') 
sit = max_pool_2d(sit, 2, strides=2)

sit = conv_2d(sit, 112, 3, activation='relu') 
sit = conv_2d(sit, 112, 3, activation='relu') 
sit = max_pool_2d(sit, 2, strides=2) 

sit = conv_2d(sit, 224, 3, activation='LeakyReLU') 
sit = conv_2d(sit, 224, 3, activation='LeakyReLU') 
sit = max_pool_2d(sit, 2, strides=2) 
 
sit = fully_connected(sit, 784, activation='LeakyReLU') 
sit = dropout(sit, 0.8) 
sit = fully_connected(sit, 10, activation='softmax') 


sit = regression(sit, optimizer='adam', 
                     loss='categorical_crossentropy', 
                     learning_rate=0.0001) 

# Training 
model = tflearn.DNN(sit, checkpoint_path='model_MNIST', 
                    max_checkpoints=1, tensorboard_verbose=0) 

model.fit(X, Y, n_epoch=5, validation_set = 0.1, shuffle=True, 
          show_metric=True, batch_size=100, snapshot_step=20, 
          snapshot_epoch=False, run_id='MNIST')  