#ifndef NN_H
#define NN_H
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

enum activeFunction{
    SIGMOID = 1,
    RELU,
    TANH,
    SOFTMAX,
};

//Custom structure for dataset
struct dataset{
    dataset(std::vector<std::vector<double>> t_data, std::vector<double> t_answers) : 
            data(t_data), answers(t_answers){};
    std::vector<std::vector<double>> data;
    std::vector<double> answers;
};

//PATH - path to .csv file, ANS_COUNT - number of values on the output layer
dataset loadData(std::string PATH, unsigned ANS_COUNT);

//Overload for vector<> printing
template <typename T>
std::ostream& operator<<(std::ostream &os, std::vector<T> &values) {
    os << '[';
    for (unsigned i = 0; i < values.size(); i++){
        os << values[i];
        if (i != values.size() - 1) os << ", ";
    }
    os << ']';
    return os;
}

class NeuralNetwork{
public:
    NeuralNetwork();

    //Adding layer in NN
    void addLayer(unsigned neurons, activeFunction activeFunc);

    //Setting additional parameters for Network
    void compile(double trainRate_t, double alpha_t, double epochs, bool bias);

    //View a model
    void print();

    //Printing the results of using NN
    void output();

    //Train
    void fit(std::vector<std::vector<double>> *data, std::vector<double> *answers);

    //Running...
    void feedForward(std::vector<double> *data);

    //Getting output value
    std::vector<double>* getOut();

private:

    //Switch of Activation Funcions
    double actFunc(double arg, activeFunction f);

    //Derivatives of Activation Functions
    double func_deriv(double arg, activeFunction f);

    //Setting random base weights
    void setWeights();

    //Mean Square Error
    double MSE(std::vector<double> *Ytrue, std::vector<double> *Ypred);

    //* {num of Layers, {neurons on layer, layer activ_function}}
    std::pair<int, std::vector<std::pair<int, activeFunction>>> network = {0, {}};
    //* Weights of axons || Values of neurons in each layer
    std::vector<std::vector<double>> weights, values;
    double trainRate = 1, alpha = 1, epochs = 500;
    bool bias = 0;
};

#endif