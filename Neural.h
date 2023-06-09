#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
using namespace std;

//Neural network class
class NeuralNetwork {
public:
    //Construcor. setting layers, weights and values vectors
    NeuralNetwork();

    //Setting up a network
    void setup();

    //Function to view network internals
    void print();

    //Printing output of the network
    vector<double>& output();

    //Activation funcion
    double sigm(double arg);

    //Derivative of sigm
    double sigm_deriv(double arg);

    //Feed forward function
    void feedForward(vector<double>* data);

    //Setting base weights
    void setWeights();

    //Mean Square Error
    double MSE(double Ytrue);

    //Training function
    void train(vector<vector<double>>* data, vector<double>* answers);

private:
    //* Layers, neurons
    pair<int, vector<int>> network;
    //* Weights of axons
    vector<vector<double>> weights;
    //* Values of neurons in each layer
    vector<vector<double>> values;
    double trainRate = 1, alpha = 1;
    int epochs = 500;
};

#endif