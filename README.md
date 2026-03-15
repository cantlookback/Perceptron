# Simple Neural Network (Perceptron) in C++

A lightweight implementation of a feedforward neural network written from scratch in C++, without external machine learning frameworks.

The project demonstrates how a neural network can be implemented using basic linear algebra and backpropagation.

## Features

### Neural Network

- Fully connected feedforward neural network
- Support for multiple hidden layers
- Configurable number of neurons per layer
- Bias neuron (can be enabled or disabled)
- Mini-batch training

### Activation Functions

Implemented activation functions:

- Sigmoid
- ReLU
- Tanh
- Softmax

### Loss Functions

- Mean Squared Error (MSE)
- Categorical Crossentropy

### Dataset Handling

- Loading datasets from .csv files
- Automatic data normalization
- Automatic train/test split
- `dataset` structure for convenient storage of:
  - training data
  - training labels
  - test data
  - test labels

### Configuration via JSON

The neural network architecture and training parameters are defined in a config.json file.

This allows changing:

- network architecture
- activation functions
- training parameters
- dataset path

without recompiling the program.

## Project Structure

```text
Perceptron/
│
├── main.cpp        # Program entry point
├── NN.h            # Neural network class
├── NN.cpp          # Neural network implementation
├── NN_utils.h      # Utility functions and dataset structures
├── NN_utils.cpp    # Dataset loading, config parsing, helpers
│
├── config.json     # Neural network configuration
│
└── data/
    ├── iris.csv
    ├── xor.csv
    └── diabetes.csv
```

## Example Configuration (config.json)

```json
{
    "dataset": {
        "filepath": "./data/iris.csv",
        "answerSize": 1,
        "classesCount": 3
    },
    "network": {
        "bias": true,
        "trainRate": 0.7,
        "alpha": 0.1,
        "epochs": 3,
        "batch_size": 1,
        "loss": "crossentropy"
    },
    "layers": [
        {
            "neurons": 4,
            "activation": "sigmoid"
        },
        {
            "neurons": 3,
            "activation": "softmax"
        }
    ]
}
```

## Build

Compile using g++:

    g++ -O3 main.cpp NN.cpp NN_utils.cpp -o main

On Windows:

    g++.exe -O3 main.cpp NN.cpp NN_utils.cpp -o main.exe

The exact command may differ depending on the compiler you use.

## Running

Run the program:

    ./main.exe

The program will:

1. Load the configuration from `config.json`
2. Load the dataset from the specified `.csv` file
3. Build the neural network
4. Train the model
5. Test the model on the test dataset
6. Print the final accuracy

## Included Datasets

Datasets included in the `/data` folder:

- Iris dataset
- XOR dataset
- Diabetes dataset

## Educational Purpose

This project is intended for learning and experimentation.

The goal is to demonstrate:

- forward propagation
- backpropagation
- gradient descent
- neural network training without external libraries
