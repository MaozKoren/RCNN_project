import random

from keras.datasets import mnist
#import tensorflow
from tensorflow.compat import v2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import SGD

random.seed(3)

randomlist = []
for i in range(0,5):
    n = random.randint(1,30)
    randomlist.append(n)

(train_X, train_y), (test_X, test_y) = mnist.load_data()

for i in randomlist:
    cv2.imshow('canvasOutput', train_X[i]);
    cv2.waitKey(0)
    cv2.destroyAllWindows()

train_X_norm = cv2.normalize(train_X, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# train_y_norm = cv2.normalize(train_y, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
test_X_norm = cv2.normalize(test_X, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# test_y_norm = cv2.normalize(test_y, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

cv2.imshow('canvasOutput', train_X_norm[0]);
cv2.waitKey(0)
cv2.destroyAllWindows()

# creating instance of labelencoder
labelencoder = OneHotEncoder(handle_unknown='ignore')
# Assigning numerical values and storing in another column
train_y_encoded = labelencoder.fit_transform(train_y.reshape(-1, 1))
test_y_encoded = labelencoder.fit_transform(test_y.reshape(-1, 1))

def create_model(nodes_1,nodes_2):
	# create model
	model = Sequential()
	model.add(Dropout(0.2, input_shape=(60,)))
	model.add(Dense(nodes_1, activation='relu', kernel_constraint=MaxNorm(3)))
	model.add(Dense(nodes_2, activation='relu', kernel_constraint=MaxNorm(3)))
    # model.add(Dense(60, activation='relu', kernel_constraint=MaxNorm(3)))
    # model.add(Dense(30, activation='relu', kernel_constraint=MaxNorm(3)))
	# Compile model
	sgd = SGD(learning_rate=0.1, momentum=0.9)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(model=create_model, epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Visible: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))