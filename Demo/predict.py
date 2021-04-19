'''
FMNet prediction:
This code uses the well-trained network model named "fmnet.cnn".
It will perform the prediction for the three angles of a focal mechanism.
A small test dataset is provided. You can directly run the model on the small test dataset.
'''

from __future__ import print_function
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import keras
from keras.datasets import mnist
from keras import losses
from keras import optimizers
from keras.models import Sequential
from keras.models import Sequential,load_model, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv1D,Conv2D,MaxPooling1D,MaxPooling2D,BatchNormalization
from keras.layers import UpSampling1D,UpSampling2D,MaxPooling1D,MaxPooling2D 
from keras.layers import Cropping1D,Cropping2D,ZeroPadding1D,ZeroPadding2D 
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.utils import plot_model
import scipy.stats as stats
import read2
import scipy.io as sio
import numpy as np
import time
import sys
import math
np.random.seed(7)

test_step=1;

xm,xn,x_test,ym,yn,y_test=read2.load_data(sgynam='test_data',sgyf1=1,sgyt1=1,step1=test_step,sgyf2=1,sgyt2=1,step2=1)

print(x_test.shape)
print(y_test.shape)

# input image dimensions
img_rows, img_cols = xm, xn
out_rows, out_cols = ym, yn

# the data, shuffled and split between train and test sets
x_test  =   x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
y_test  =   y_test.reshape(y_test.shape[0], 1, out_rows, out_cols)


x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

model = load_model('fmnet.cnn')
y_pred = model.predict([x_test])

print(type(y_test))
print(type(y_pred))
print(y_test.shape)
print(y_pred.shape)

output_folder = "test_output"
if not os.path.exists(output_folder):
      os.mkdir(output_folder)

for i in range(y_pred.shape[0]):
      out_test=(y_test[i])
      out_pred=(y_pred[i])
      filename="test_output/predict_%06d.mat" %(i+1)
      sio.savemat(filename,{'true':out_test,'pred':out_pred})
