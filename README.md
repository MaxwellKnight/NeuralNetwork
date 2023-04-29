# Multi-Layer Perceptron (MLP) Library in C++

This is a simple library for building and training multi-layer perceptron (MLP) models in C++. The library provides basic functionalities such as defining the network architecture, parsing input data, forward and backward propagation, and updating the weights.

## Getting Started

Prerequisites
To use the MLP library, you need to have a C++ compiler installed on your system. The code has been tested on GCC 9.3.0.

## Installing

To install the MLP library, you can simply clone the repository to your local machine:

bash
Copy code
git clone https://github.com/MaxwellKnight/NeuralNetwork.git
Alternatively, you can download the code as a ZIP file and extract it to a directory of your choice.

## Usage

To use the MLP library, you need to include the header files MLP.h, neuron.h, and scalar.h in your code. The main.cpp file provides an example of how to use the library to train an MLP model on a simple dataset.

### The basic steps for building an MLP model using the library are:

1 - Define the network architecture by specifying the number of inputs and neurons in each layer. This is done by creating a std::vector<int> object dims, where dims[i] is the number of neurons in layer i.
2 - Create an instance of the NeuralNetwork<T> class, where T is the type of the input and output data (e.g., double).
3 - Parse the input data into a std::vector<std::vector<scalar<T>\*>> object using the parse_input() function.
4 - Train the model by iterating over the dataset for a fixed number of epochs. In each epoch, perform a forward pass on the input data to get the predicted output, calculate the loss using a suitable loss function (e.g., mean squared error), and perform a backward pass to update the weights using gradient descent.
The main.cpp file provides an example of how to perform these steps in code. You can modify the code to suit your needs.

### Contributing

Contributions to the MLP library are welcome. If you find a bug or have a feature request, please open an issue on the GitHub repository.
