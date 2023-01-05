import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import seaborn as sns
#from plotnine import *
from tensorflow.keras.layers import Normalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

mtcars = pd.read_csv('mtcars.csv')
# This needs to be added in front, or the solution will be very unstable
normalizer = Normalization(input_shape=[1,], axis=None)
normalizer.adapt(mtcars['hp'])

mtcars['hp_norm'] = np.array(normalizer(mtcars['hp'])).flatten()
train_data, test_data = train_test_split(mtcars, test_size=0.1)

def create_poly_features(series, n):
    poly = PolynomialFeatures(n)
    return poly.fit_transform(np.expand_dims(series, axis=1))

n = 2  # Use quadratic polynomial regression
model = keras.Sequential([
    layers.Input(shape=(n+1,)),
    layers.Dense(units=1)
])

model.compile(optimizer=Adam(learning_rate=0.1),loss='mean_squared_error')

history = model.fit(
    create_poly_features(train_data['hp_norm'], n), train_data['mpg'],
    epochs=100, verbose=0, validation_split = 0.2)

prediction_results = pd.DataFrame({'network_b' : model.predict(create_poly_features(test_data['hp_norm'], n)).flatten(),
                                   'reference mpg' : test_data['mpg']})
print(prediction_results)