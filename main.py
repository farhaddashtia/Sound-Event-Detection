import theano
print (theano.__version__)
import tensorflow as tf
import keras
print (keras.__version__)	
import numpy as np
np.random.seed(123)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Merge,LSTM,TimeDistributed
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils


from keras import backend as K
K.set_image_dim_ordering('th')


import scipy.io
mat = scipy.io.loadmat('data4mynet2.mat')
X_train_GCC=mat['GCC_train']
X_test_GCC=mat['GCC_test']
X_train_mel=mat['MEL_E_train']
X_test_mel=mat['MEL_E_test']
X_train_ACR=mat['ACR_train']
X_test_ACR=mat['ACR_test']

Y_train=mat['train_labels2']
Y_test=mat['test_labels2']


T=192
left_branch= Sequential() 
left_branch.add(Convolution2D(100, 3, activation='relu', input_shape=(T,60,3),data_format="channels_last"))
left_branch.add(BatchNormalization())
left_branch.add(MaxPooling2D(pool_size=(3,1)))
left_branch.add(Dropout(0.5))
left_branch.add(Convolution2D(100, 3, activation='relu'))
left_branch.add(BatchNormalization())
left_branch.add(MaxPooling2D(pool_size=(2,1)))
left_branch.add(Dropout(0.5))
left_branch.add(Convolution2D(100, 3, activation='relu'))
left_branch.add(BatchNormalization())
left_branch.add(MaxPooling2D(pool_size=(2,1)))
left_branch.add(Dropout(0.5))

middle_branch= Sequential() 
middle_branch.add(Convolution2D(100, 3, activation='relu', input_shape=(T,40,2),data_format="channels_last"))
middle_branch.add(BatchNormalization())
middle_branch.add(MaxPooling2D(pool_size=(2,1)))
middle_branch.add(Dropout(0.5))
middle_branch.add(Convolution2D(100, 3, activation='relu'))
middle_branch.add(BatchNormalization())
middle_branch.add(MaxPooling2D(pool_size=(2,1)))
middle_branch.add(Dropout(0.5))
middle_branch.add(Convolution2D(100, 3, activation='relu'))
middle_branch.add(BatchNormalization())
middle_branch.add(MaxPooling2D(pool_size=(2,1)))
middle_branch.add(Dropout(0.5))

right_branch= Sequential() 
right_branch.add(Convolution2D(100, 3, activation='relu', input_shape=(T,400,2),data_format="channels_last"))
right_branch.add(BatchNormalization())
right_branch.add(MaxPooling2D(pool_size=(10,1)))
right_branch.add(Dropout(0.5))
right_branch.add(Convolution2D(100, 3, activation='relu'))
right_branch.add(BatchNormalization())
right_branch.add(MaxPooling2D(pool_size=(4,1)))
right_branch.add(Dropout(0.5))
right_branch.add(Convolution2D(100, 3, activation='relu'))
right_branch.add(BatchNormalization())
right_branch.add(MaxPooling2D(pool_size=(2,1)))
right_branch.add(Dropout(0.5))

#print(right_branch.output_shape)
#print(middle_branch.output_shape)
#print(left_branch.output_shape)

merged1=keras.layers. Merge([left_branch, middle_branch,right_branch], mode='concat')
#print(merged1.output_shape)

lstm_l=Sequential()
lstm_l.add(merged1)
lstm_l.add(TimeDistributed(Flatten()))
#print(lstm_l.output_shape)
lstm_l.add(LSTM(100, activation='tanh', dropout=0.5, go_backwards=False))

lstm_r=Sequential()
lstm_r.add(merged1)
lstm_r.add(TimeDistributed(Flatten()))
#print(lstm_r.output_shape)
lstm_r.add(LSTM(100, activation='tanh', dropout=0.5, go_backwards=True))

merged2=keras.layers. Merge([lstm_l,lstm_r], mode='concat')

num_class=7;
final_model = Sequential()
final_model.add(merged2)
final_model.add(Dense(num_class, activation='sigmoid'))
print(final_model.output_shape)
final_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])



final_model.fit([X_train_GCC, X_train_mel, X_train_ACR], Y_train,  batch_size=32, nb_epoch=2, verbose=1)

score =final_model.evaluate([X_test_GCC, X_test_mel, X_test_ACR], Y_test, verbose=0)
