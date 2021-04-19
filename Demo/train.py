'''
FMNet training:
This code shows the network architecture of our FMNet.
Training data is not provided since it is too large.
You need specify your own training data path if you want to retrain the model.
We have provided the well-trained model named "fmnet.cnn".
'''

from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
import keras
from keras.datasets import mnist
from keras import losses
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv1D,Conv2D,MaxPooling1D,MaxPooling2D,BatchNormalization
from keras.layers import UpSampling1D,UpSampling2D,MaxPooling1D,MaxPooling2D 
from keras.layers import Cropping1D,Cropping2D,ZeroPadding1D,ZeroPadding2D 
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.utils import plot_model
import scipy.stats as stats
import read2
import numpy as np
import time
import sys
import math
np.random.seed(7)

train_step=1;
test_step=100;

xm,xn,x_train,ym,yn,y_train=read2.load_data(sgynam='PLEASE INPUT YOUR TRAINING DATA PATH',sgyf1=1,sgyt1=100000,step1=train_step,sgyf2=1,sgyt2=1,step2=1)
xm,xn,x_test,ym,yn,y_test=read2.load_data(sgynam='PLEASE INPUT YOUR TRAINING DATA PATH',sgyf1=1,sgyt1=100000,step1=test_step,sgyf2=1,sgyt2=1,step2=1)

batch_size = 16
epochs = 50
  
# input image dimensions
img_rows, img_cols = xm, xn
out_rows, out_cols = ym, yn

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
y_train = y_train.reshape(y_train.shape[0], 1, out_rows, out_cols)
x_test  =   x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
y_test  =   y_test.reshape(y_test.shape[0], 1, out_rows, out_cols)

main_input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

main_input = Input(shape=main_input_shape,name='main_input')
x=Conv2D(64, kernel_size=(3,3),padding='same')(main_input)
x=MaxPooling2D(pool_size=(2,2),padding='same')(x) # 1
x=LeakyReLU(alpha=0.2)(x)
x=BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones')(x)
x=Conv2D(64, kernel_size=(3,3),padding='same')(x)
x=MaxPooling2D(pool_size=(2,2),padding='same')(x) #2
x=LeakyReLU(alpha=0.2)(x)
x=BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones')(x)
x=Conv2D(64, kernel_size=(3,3),padding='same')(x)
x=MaxPooling2D(pool_size=(2,2),padding='same')(x) #3
x=LeakyReLU(alpha=0.2)(x)
x=BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones')(x)
x=Conv2D(64, kernel_size=(3,3),padding='same')(x)
x=MaxPooling2D(pool_size=(2,2),padding='same')(x) #4
x=LeakyReLU(alpha=0.2)(x)
x=BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones')(x)
x=Conv2D(64, kernel_size=(3,3),padding='same')(x)
x=MaxPooling2D(pool_size=(3,2),padding='same')(x) #5
x=LeakyReLU(alpha=0.2)(x)
x=BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones')(x)
x=Conv2D(128, kernel_size=(1,3),padding='same')(x)
x=MaxPooling2D(pool_size=(1,2),padding='same')(x) #6
x=LeakyReLU(alpha=0.2)(x)
x=BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones')(x)
x=Conv2D(128, kernel_size=(1,3),padding='same')(x)
x_encode=MaxPooling2D(pool_size=(1,2),padding='same')(x) #7
print(x_encode.shape)
x=LeakyReLU(alpha=0.2)(x_encode)
x=BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones')(x)
x=Conv2D(128, kernel_size=(1,1),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #1
x=LeakyReLU(alpha=0.2)(x)
x=BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones')(x)
x=Conv2D(128, kernel_size=(1,3),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #2
x=LeakyReLU(alpha=0.2)(x)
x=BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones')(x)
x=Conv2D(64, kernel_size=(1,3),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #3
x=LeakyReLU(alpha=0.2)(x)
x=BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones')(x)
x=Conv2D(64, kernel_size=(1,3),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #4
x=LeakyReLU(alpha=0.2)(x)
x=BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones')(x)
x=Conv2D(64, kernel_size=(1,3),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #5
x=LeakyReLU(alpha=0.2)(x)
x=BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones')(x)
x=Conv2D(32, kernel_size=(1,3),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #6
x=LeakyReLU(alpha=0.2)(x)
x=BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones')(x)
x=Conv2D(32, kernel_size=(1,3),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #7
x=LeakyReLU(alpha=0.2)(x)
x=BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones')(x)
x=Conv2D(16, kernel_size=(1,3),padding='same')(x)
x=LeakyReLU(alpha=0.2)(x)
x=BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones')(x)
main_output=Conv2D(3, kernel_size=(1,3),padding='same')(x)
print(main_output.shape)

model = Model(inputs=[main_input],outputs=[main_output])
model_encode = Model(inputs=[main_input],outputs=[x_encode])
optimizer = keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08)

model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['accuracy'])

history_callback=model.fit([x_train], 
                           [y_train],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=([x_test], [y_test]))

# Open the save commond when you retrain the model
#model.save('fmnet.cnn')
#model_encode.save('encode.cnn')
