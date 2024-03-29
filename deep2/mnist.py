import pickle

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os

from tensorflow.python.keras.utils.np_utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Data Loading
from tensorflow.keras.datasets.mnist import load_data;
(x_train,y_train),(x_test,y_test) = load_data(path='mnist.npz');

print(x_train.shape,y_train.shape)
print(x_train)
print('-----------------------------')
print(y_train)

#show image
# import matplotlib.pyplot as plt
# img = x_train[0,:];
# label = y_train[0];
# plt.figure();
# plt.imshow(img)
# plt.title('%d %d' % (0,label), fontsize = 15);
# plt.show()

# 데이터 정규화
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import train_test_split;

x_train, x_vali , y_train, y_vali = train_test_split(x_train,y_train,
                                                      test_size=0.3,
                                                     random_state=777
                                                     );
print(x_train.shape,y_train.shape)
print(x_vali.shape,y_vali.shape)
x_train = x_train.reshape(x_train.shape[0],28*28)
x_vali = x_vali.reshape(x_vali.shape[0],28*28)
x_test = x_test.reshape(x_test.shape[0],28*28)

x_train_scaler = MinMaxScaler().fit_transform(x_train)
x_test_scaler = MinMaxScaler().fit_transform(x_test)
x_vali_scaler = MinMaxScaler().fit_transform(x_vali)

print(x_train_scaler[0,:])

#데이터를 범주형으로 변환
print(y_train.shape)
print(y_train[1])
y_train = to_categorical(y_train)
print(y_train.shape)
print(y_train[1])
y_vali = to_categorical(y_vali)
y_test = to_categorical(y_test)

#모델 구성 - 신경망 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense;

model = Sequential();
model.add(Dense(128, activation='relu',input_shape=(784,)))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(x_train_scaler,y_train,
                    epochs=10,
                    validation_data=(x_vali_scaler, y_vali)
                    ,verbose=0)

print(model.evaluate(x_test_scaler,y_test))

# with open('mnist.model',"wb") as w:
#     pickle.dump(model,w)

model.save("mnist.h5")
# result = model.predict(x_test_scaler)
# print(result.shape)
# print(result[0])
#
# arg_result = np.argmax(result, axis=1)
#
# plt.imshow(x_test[0].reshape(28,28));
# plt.title(str(arg_result[0]))
# plt.show()
