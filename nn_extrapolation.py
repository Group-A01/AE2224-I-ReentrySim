#!/usr/bin/env python3

# Multi-layer perceptron regression, lab session 2 of AE-2224-II:
# By Guido de Croon

import numpy as np
import torch
from matplotlib import pyplot as plt

from scipy import signal

from alive_progress import alive_bar
# import load_data
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train_regressor_nn(n_features, n_hidden_neurons, learning_rate, n_epochs, X, Y):
    # Define the model:
    
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, n_hidden_neurons),
        # torch.nn.Sigmoid(),
        torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
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
    data = np.genfromtxt("ap_initial_data.txt", delimiter=" ")
    X = data[:,0]
    Y = data[:,1]
    X = X[:200]
    Y = Y[:200]
else:
    X = np.linspace(0,5,200)
    Y = np.sin(X)

# Total number of samples:
n_features = 1

#savgol filter
# X = signal.savgol_filter(X, 70, 1, axis=0) 

#convert to torch tensors
X = torch.from_numpy(X).float().view(-1, 1)
Y = torch.from_numpy(Y).float().view(-1, 1)

# Make a neural network model for the MLP with sigmoid activation functions in the hidden layer, and linear on the output
n_hidden_neurons = 64
# rates = np.linspace(0.04, 0.04200424116196333, 3)
# rates = [0.04200424116196333]
rates = np.linspace(1e-5, 1e-1, 5)
n_epochs = 2000

errors = []
for i in rates:
    learning_rate = i
    plt.subplot(3,1,1)
    plt.suptitle("Current learning rate: {0}".format(learning_rate))
    plt.plot(X, Y, label="Actual")
    plt.savefig('data.png')

    print("Current learning rate: {0}".format(learning_rate))
    model = train_regressor_nn(n_features, n_hidden_neurons, learning_rate, n_epochs, X, Y)

    # plot the output of the network vs. the ground truth:
    y_pred = model(X)
    y_pred = y_pred.detach().numpy()
    y_plot = Y.detach().numpy()
    # y_plot = y_plot.reshape(N, 1)

    print('Root Mean Squared Error: ', np.sqrt(np.mean((y_pred - y_plot)**2)))
    errors.append(np.sqrt(np.mean((y_pred - y_plot)**2)))
    plt.subplot(3,1,2)
    plt.plot(y_pred, label="LR = {0}".format(learning_rate), linewidth=3)
    plt.legend()
    
    # inp = input("show?")
    # if(inp=="1"):
    #     plt.show()
    # plt.clf()
    # plt.savefig('output_vs_ground_truth.png')

plt.plot(y_plot,'ro', label='Ground Truth')

plt.subplot(3,1,3)
plt.plot(rates, errors)
plt.xscale("log")

plt.show()
