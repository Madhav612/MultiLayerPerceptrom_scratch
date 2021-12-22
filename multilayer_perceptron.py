# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding


class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden = 16, hidden_activation = 'sigmoid', n_iterations = 1000, learning_rate = 0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        # print(hidden_activation)
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = 100
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None

    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        
        self._X = X
        self._y = one_hot_encoding(y)#call one-hot encoding method to convert the target variable into one-hot encoding 
        
        self._h_weights = np.random.randn(self._X.shape[1],self.n_hidden) * np.sqrt(2/self.n_hidden)#https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
        self._o_weights = np.random.randn(self.n_hidden,len(self._y[0])) * np.sqrt(2/len(self._y[0]))#h1 initialization method
        self._h_bias = np.zeros((1, self.n_hidden))#hidden layer bias would have dimension (1,hidden_layer_size)
        self._o_bias = np.zeros((1,len(self._y[0])))#bias term would have number of classes in the target variable as the bias term  length

        np.random.seed(42)

        # raise NotImplementedError('This function must be implemented by the student.')

    #The appproach below was considered from this video: https://www.youtube.com/watch?v=w8yWXqWQYmU

    def fit(self, X, y):
        self._initialize(X, y)
        
        for i in range(self.n_iterations):#doing n_iterations    
            z1,a1,z2,a2 = self.forward()#calling forward method to calculate forward propogation
            cross_entropy_loss = self.backward(X,self._y,z1,a1,z2,a2)#calling backward method to calculate backward propogation
            if (i%20==0) and i!=0:#storing the loss every 20 iterations
                self._loss_history.append(cross_entropy_loss)


        # raise NotImplementedError('This function must be implemented by the student.')
    
    def forward(self):
        a0 = self._X
        z1 = np.dot(a0,self._h_weights) + self._h_bias#a0 is our input and h_weights is hidden layer weights
        a1 = self.hidden_activation(z1)#passing the above variable to activation function
        z2 = np.dot(a1,self._o_weights) + self._o_bias
        a2 = self._output_activation(z2)#softmax activation applied
        return z1,a1,z2,a2

    def backward(self,X,y,z1,a1,z2,a2):
        cross_entropy_loss = np.sum(cross_entropy(y,a2))#calculating the cross entropy using the target variable and output of softmax activation function
        a2_derivative = (a2 - y) * self._output_activation(z2,derivative=True)# derivative(change in values) total error multiplied by derivative of the softmax function 
        z2_derivative = np.dot(a2_derivative,self._o_weights.T)#z2 is calculated by taking the dot product of the above value and output weights
        a1_derivative = z2_derivative * self.hidden_activation(z1,derivative=True)#derivative(change in values) is calculated by taking the derivative of activation function used for forward propogation

        self._o_weights -= self.learning_rate * np.dot(a1.T,a2_derivative)#we update the output layer weights 
        self._o_bias -= self.learning_rate * np.sum(a2_derivative,axis=0,keepdims=True)#update output layer bias
        self._h_weights -= self.learning_rate * np.dot(self._X.T,a1_derivative)#update the hidden layer weights
        self._h_bias -= self.learning_rate * np.sum(a1_derivative,axis=0,keepdims=True)#update the hidden layer bias
        return cross_entropy_loss
    
    #Approach considered ended

    def predict(self, X):
        self._X = X
        # print(self._y)
        z1,a1,z2,a2 = self.forward()#only calling forward propogation
        predicted = []
        for j in range(len(a2)):
            predicted.append(a2[j].argmax())
        # print(predicted)
        return np.array(predicted)

        # raise NotImplementedError('This function must be implemented by the student.')
