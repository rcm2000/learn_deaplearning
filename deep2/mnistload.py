import pickle

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os

from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

loaded_model = keras.models.load_model("mnist.h5")
print('completed load.........')