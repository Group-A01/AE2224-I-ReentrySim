#!/usr/bin/env python3

# Multi-layer perceptron regression, lab session 2 of AE-2224-II:
# By Guido de Croon

import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd

from scipy import signal

from alive_progress import alive_bar
# import load_data
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train_regressor_nn(n_features, n_hidden_neurons, learning_rate, n_epochs, X, Y):
    # Define the model:
    
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, n_hidden_neurons),
        torch.nn.Sigmoid(),
        # torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
        # torch.nn.Sigmoid(),
        torch.nn.Linear(n_hidden_neurons, 1)
    )

    # MSE loss function:
    loss_fn = torch.nn.MSELoss()

    # optimizer:
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=learning_rate, 
                                #  momentum=0.9
                                 )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.0001)

    # Train the network:
    with alive_bar(n_epochs) as bar:
        for i in range(n_epochs):
            # print("Epoch: {0}".format(i))
            y_pred = model(X)
            loss = loss_fn(y_pred, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            bar()

    # return the trained model
    return model


switch = True
if switch:
    # data = np.genfromtxt("snfuture.csv", delimiter=",")
    data = pd.read_csv("real_data.csv")
    print(data)
    X = data.loc[:,["days", "dB", "SN"]]
    Y = data.loc[:,["Ap"]]
    slce = 0.5
    sample = 1000
    
    X_train = X[:int(sample*slce)] #train data
    Y_train = Y[:int(sample*slce)]

    X_test = X[int(sample*slce):sample]
    Y_test = Y[int(sample*slce):sample]
    
    print(X_train, X_train.shape)
    print(X_test, X_test.shape)
    print(Y_train, Y_train.shape)
    print(Y_test, Y_test.shape)
else:
    X = np.linspace(0,5,200)
    Y = np.sin(X)

# Total number of samples:
n_features = X_train.shape[1]

#savgol filter
# X = signal.savgol_filter(X, 70, 1, axis=0) 
data.plot(x="days", title="Data")
# plt.show()
# plt.plot(X, Y, label="Actual")
# plt.savefig('data.png')

#convert to torch tensors
X_train = torch.tensor(X_train.values).float()
Y_train = torch.tensor(Y_train.values).float()
X_test = torch.tensor(X_test.values).float()
Y_test = torch.tensor(Y_test.values).float()
# X = torch.from_numpy(X).float()
# Y = torch.from_numpy(Y).float()

# Make a neural network model for the MLP with sigmoid activation functions in the hidden layer, and linear on the output
n_hidden_neurons = 64
# rates = np.linspace(0.04, 0.04200424116196333, 3)
# rates = [0.04200424116196333]
rates = np.linspace(0.01, 0.025, 1)
n_epochs = 2000
plt.figure()
errors = []
for i in rates:
    learning_rate = i
    plt.suptitle("Current learning rate: {0}".format(learning_rate))
    print("Current learning rate: {0}".format(learning_rate))
    model = train_regressor_nn(n_features, n_hidden_neurons, learning_rate, n_epochs, X_train, Y_train)

    # plot the output of the network vs. the ground truth:
    y_pred = model(X_test)
    y_used = model(X_train)
    y_used = y_used.detach().numpy()
    y_pred = y_pred.detach().numpy()
    y_plot = Y_test.detach().numpy()
    y_train = Y_train.detach().numpy()
    # y_plot = y_plot.reshape(N, 1)

    print('Root Mean Squared Error: ', np.sqrt(np.mean((y_pred - y_plot)**2)))
    errors.append(np.sqrt(np.mean((y_pred - y_plot)**2)))
    plt.subplot(2,1,1)
    plt.plot(X_train[:,0], y_used, linewidth=3)
    plt.plot(X_test[:,0], y_pred, label="LR = {0}".format(learning_rate), linewidth=3)
    plt.legend()

plt.plot(X_train[:,0], y_train,'r-', label='Train Set')
plt.plot(X_test[:,0], y_plot,'b-', label='Test Set')

plt.subplot(2,1,2)
plt.plot(rates, errors)
plt.xscale("log")

plt.show()
