import numpy as np
import pandas as pd
import os

from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATA_PATH = './csv_data/nocolorinfo'

train_df = pd.read_csv(DATA_PATH + '/train.csv')
val_df = pd.read_csv(DATA_PATH + '/val.csv')
test_df = pd.read_csv(DATA_PATH + '/test.csv')

class_col = ['black', 'blue', 'brown', 'green', 'red', 'white',
             'dress', 'shirt', 'pants', 'shorts', 'shoes']

#print(train_df.head())

# 신경망 구축
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
model = Sequential()

# 입력 데이터의 형태를 꼭 명시해야 합니다.
model.add(Flatten(input_shape = (112, 112, 3))) # (112, 112, 3) -> (112 * 112 * 3)
model.add(Dense(128, activation = 'relu')) # 128개의 출력을 가지는 Dense 층
model.add(Dense(64, activation = 'relu')) # 64개의 출력을 가지는 Dense 층
model.add(Dense(11, activation = 'sigmoid')) # 11개의 출력을 가지는 신경망

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['acc'])

# Make Generator
# 이미지 제네레이터를 정의합니다.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

batch_size = 32

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col = 'image',
    y_col = class_col,
    target_size = (112, 112),
    color_mode='rgb',
    class_mode='raw',
    batch_size=batch_size,
    shuffle = True,
    seed=42
)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=None,
    x_col = 'image',
    y_col = class_col,
    target_size = (112, 112),
    color_mode='rgb',
    class_mode='raw',
    batch_size=batch_size,
    shuffle=True
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col = 'image',
    y_col = None,
    target_size = (112, 112),
    color_mode='rgb',
    class_mode=None,
    batch_size=batch_size,
    shuffle = False
)

# 학습
model.fit(train_generator,
         validation_data = val_generator,
         epochs = 10, verbose=1)

# model 저장
model.save("p121.h5")
loaded_model = keras.models.load_model("p121.h5")

# 테스트 데이터를 이용한 예측
result = loaded_model.predict(test_generator)

import matplotlib.pyplot as plt
import cv2  # pip install opencv-python

image = cv2.imread(test_df['image'][50])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image);

desc = zip(class_col,list(result[50]))
desc_list = list(desc);
color = desc_list[0:6];
type = desc_list[6:11];
type = sorted(type, key=lambda z:z[1], reverse=True)
color = sorted(color, key=lambda z:z[1], reverse=True)

print(type[0][0],type[0][1])
print(color[0][0], color[0][1])

plt.title(type[0][0]+' '+str(type[0][1])+' '+color[0][0]+' '+str(color[0][1]));
plt.show();