## About the project
This exercise demonstrates how a simple n layer neural network can be build from scratch in the R programming language.
The aim of this exercise was to develop a fundamental intuition for the back propagation algorithm and gradient descent.

## Features and usage
The dan_net.R script contains a single function which takes the following inputs:
* **x** - training examples which may consist of multiple columns/factors
* **y** - response examples corresponding to the independent observations in x
* **hidden layers** - a vector representing the number of nodes the user wishes to use in each hidden layer. For example, the input [3,3,2] would generate a network architecture consisting of 3 hidden layers with the first containing 3 nodes, the next containing 3 nodes, and the last containing 2 nodes.
* **cost function** - the loss function the user wishes to implement. for example, sum of squared error (SSE) or cross entropy (WIP) 
* **learning rate** - the learning rate 'alpha' the network should adopt
* **learning cycles** - the number of learning iterations the network should adopt
