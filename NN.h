#ifndef NN_H
#define NN_H
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

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
    void addLayer(unsigned neurons);

    //Setting additional parameters for Network
    void compile(uint64_t trainRate_t, uint64_t alpha_t, uint64_t epochs);

    //View a model
    void print();

    //Printing the results of using NN
    void output();

    //Train
    void fit(std::vector<std::vector<double>>* data, std::vector<double>* answers);

    //Running...
    void feedForward(std::vector<double>* data);

private:
    //Activation funcion
    double sigm(double arg);

    //Derivative of sigm
    double sigm_deriv(double arg);

    //Setting random base weights
    void setWeights();

    //Mean Square Error
    double MSE(double Ytrue);

    //* Layers, neurons
    std::pair<int, std::vector<int>> network = {0, {}};
    //* Weights of axons
    std::vector<std::vector<double>> weights;
    //* Values of neurons in each layer
    std::vector<std::vector<double>> values;
    uint64_t trainRate = 1, alpha = 1;
    uint64_t epochs = 500;
};

#endif