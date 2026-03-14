#ifndef NN_H_
#define NN_H_

#include "NN_utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork();

    // Adding layer in NN
    void addLayer(unsigned neurons, activeFunction activeFunc = activeFunction::SIGMOID);

    // Setting additional parameters for Network
    void compile(double trainRate_t=1, double alpha_t=1, double epochs_t=100, bool bias_t=0, lossFunction loss_t=lossFunction::MSE, unsigned batch_size_t=1);

    // View a model
    void print();

    // Train
    void fit(std::vector<std::vector<double>>& data, std::vector<std::vector<double>>& answers);

    // Getting prediction
    std::vector<double> predict(const std::vector<double>& input);

private:
    // Layer struct
    struct Layer {
        unsigned neurons;
        activeFunction activation;
    };

    // Running...
    void feedForward(const std::vector<double>& data);

    // Activation Funcions switch
    double actFunc(double arg, activeFunction f);

    // Derivatives of Activation Functions
    double func_deriv(double arg, activeFunction f);

    // Setting random base weights
    void setWeights();

    // Loss Functions switch
    double lossFunc(std::vector<std::vector<double>>& Ytrue, std::vector<std::vector<double>>& Ypred);

    //*Vector of NN layers
    std::vector<Layer> layers;
    //*Weights of axons || Values of neurons in each layer
    std::vector<std::vector<double>> weights, values;
    //*Hyperparameters
    double trainRate, alpha;
    unsigned epochs, batch_size;
    //*Bias marker
    bool bias;
    //*Loss function for switch
    lossFunction loss;
};

#endif  // NN_H_