# Perceptron-Classifier
This is a simple perceptron neural network to classify the MNIST handwritten numbers database using a subset of the MNIST training database in jpg format. (5000 images)

This neural netowrk uses the softmax activation function for the output neurons and the log-likelihood cost function. It also uses stochastic gradient descent with linear
reduction in learning rate and L2 regularization.

With an inital learning rate of 0.6, regularization parameter of 0.0007, and network topology with 2 hidden layers with 16 neurons each, this network achieves a
classification accuracy of 92% when trained for 10 epochs. (Tested over the whole MNIST testing database)
