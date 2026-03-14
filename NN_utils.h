#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "json.hpp"

using json = nlohmann::json;

// Colors
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
    dataset(const std::vector<std::vector<double>>& t_data, const std::vector<std::vector<double>>& t_answers, const std::vector<std::vector<double>>& t_test_data,
            const std::vector<std::vector<double>>& t_test_answers)
        : data(t_data), answers(t_answers), test_data(t_test_data), test_answers(t_test_answers) {};

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
dataset loadData(const std::string& PATH, unsigned ANS_COUNT, unsigned OUTPUT_COUNT);

void normalizeData(std::vector<std::vector<double>>* data);

void printProgress(unsigned epoch, unsigned total_epochs, double loss);

void printProgress(size_t current, size_t total);

json loadConfig(const std::string& path);

activeFunction parseActivation(const std::string& name);

lossFunction parseLoss(const std::string& name);

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

#endif