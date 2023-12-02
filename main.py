# importing our libs and moduls
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense

# prepoc'ing our data for training
dataset = pd.read_csv("city_analisys_data.csv")
dataset = np.array(dataset)

X = dataset[:, 1:]
Y = dataset[:, 0]

scaler = preprocessing.MinMaxScaler()
X_scale = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.3)

# creating a model
model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy']) # model's settings
model.fit(X_train, Y_train, batch_size=16, epochs=800) # training model
model.save('CA_model.h5') # saving model

# And, testing our model
print("\nTEST\n")
model.evaluate(X_test, Y_test)