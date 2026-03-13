#ifndef NN_H_
#define NN_H_
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

constexpr const char* RED = "\033[31m";
constexpr const char* GREEN = "\033[32m";
constexpr const char* YELLOW = "\033[33m";
constexpr const char* RESET = "\033[0m";

// Loss Function enumeration
enum class lossFunction {
    MSE = 1,
    categorical_crossentropy,
};

// Activation Functions enumeration
enum class activeFunction {
    SIGMOID = 1,
    RELU,
    TANH,
    SOFTMAX,
};

// Custom structure for dataset
struct dataset {
    dataset(const std::vector<std::vector<double>>& t_data,
            const std::vector<std::vector<double>>& t_answers,
            const std::vector<std::vector<double>>& t_test_data,
            const std::vector<std::vector<double>>& t_test_answers)
        : data(t_data),
          answers(t_answers),
          test_data(t_test_data),
          test_answers(t_test_answers) {};

    // Train part
    std::vector<std::vector<double>> data;
    std::vector<std::vector<double>> answers;
    // Test part
    std::vector<std::vector<double>> test_data;
    std::vector<std::vector<double>> test_answers;
};

// PATH - path to .csv file
// ANS_COUNT - number of answer values
// OUTPUT_COUNT - number of classes
dataset loadData(const std::string& PATH, unsigned ANS_COUNT,
                 unsigned OUTPUT_COUNT);

// Overload for vector<> printing
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& values) {
    os << '[';
    for (size_t i = 0; i < values.size(); i++) {
        os << values[i];
        if (i != values.size() - 1) os << ", ";
    }
    os << ']';
    return os;
}
class NeuralNetwork {
public:
    NeuralNetwork();

    // Adding layer in NN
    void addLayer(unsigned neurons,
                  activeFunction activeFunc = activeFunction::SIGMOID);

    // Setting additional parameters for Network
    void compile(double trainRate_t, double alpha_t, double epochs, bool bias,
                 lossFunction loss_t);

    // View a model
    void print();

    // Train
    void fit(std::vector<std::vector<double>>& data,
             std::vector<std::vector<double>>& answers);

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
    double lossFunc(std::vector<std::vector<double>>& Ytrue,
                    std::vector<std::vector<double>>& Ypred);

    //*Vector of NN layers
    std::vector<Layer> layers;
    //*Weights of axons || Values of neurons in each layer
    std::vector<std::vector<double>> weights, values;
    //*Hyperparameters
    double trainRate = 1, alpha = 1, epochs = 500;
    //*Bias marker
    bool bias = 0;
    //*Loss function for switch
    lossFunction loss;
};

#endif  // NN_H_