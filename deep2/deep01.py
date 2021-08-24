import tensorflow as tf
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)

x = [-2,3,20,0,-5,7,-23,-8,32,-15];
y = [-1,4,21,1,-4,8,-22,-7,33,-14];

x_train = np.array(x).reshape(-1,1);
y_train = np.array(y);

print(x_train)
print(y_train)
print(x_train.shape)
print(y_train.shape)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
##모델 구성
model = Sequential()
##단층 퍼셉트론 구성(신경망 구축)
model.add(Dense(units=1,activation='linear',input_dim=1))
##모델 준비
model.compile(optimizer='adam',loss='mse',metrics=['mae']);
##모델 학습
model.fit(x_train,y_train,epochs=3000,verbose=1);

x_data = [8,9,10,-1,4,-15];
x_data = np.array(x_data).reshape(-1,1)

result = model.predict(x_data);
print(result)