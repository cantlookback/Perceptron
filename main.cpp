#include <iostream>
#include "NN.h"

using namespace std;

void normalizeData(vector<vector<double>> *data){
    double maxDot = 0, minDot = (*data)[0][0];
    for (unsigned i = 0; i < (*data)[0].size(); i++){
        for (unsigned j = 0; j < data->size(); j++){
            maxDot = ((*data)[j][i]) > maxDot ? ((*data)[j][i]) : maxDot;
            minDot = ((*data)[j][i]) < minDot ? ((*data)[j][i]) : minDot;
        }

        for (unsigned k = 0; k < data->size(); k++){
            (*data)[k][i] = (((*data)[k][i]) - minDot) / (maxDot - minDot);
        }
        maxDot = 0;
        minDot = (*data)[0][0];
    }
}

int main(){
    dataset train = loadData("C:/Perceptron/data/diabetes.csv", 1, 1);
    //Iris -- 0.7, 0.1, 1000, 1 || [4, 8, 4, 1]

    unsigned INPUT_SIZE = train.data[0].size();

    normalizeData(&train.data);

    NeuralNetwork net;

    net.addLayer(INPUT_SIZE);
    net.addLayer(8, SIGMOID);
    net.addLayer(4, SIGMOID);
    net.addLayer(1, SIGMOID);

    net.compile(0.7, 0.1, 100, 1, MSE);

    net.fit(&train.data, &train.answers);

    vector<double> test;
    test.resize(INPUT_SIZE);
    while (true){
        std::cout << "Input >>";
        for (unsigned i = 0; i < INPUT_SIZE; i++){            
            std::cin >> test[i];
        }
        net.feedForward(&test);

        net.output();
    }

    //? DATA TEST MODULE
    // dataset test = loadData("C:/Perceptron/test/IrisTest3.csv", 1, 3);

    // normalizeData(&test.data);

    // for (unsigned i = 0; i < test.data.size(); i++){
    //     std::cout << "Row " << i << " testing..." << '\n';

    //     net.feedForward(&(test.data[i]));

    //     std::cout << "Got -->" << *net.getOut() << '\n';
    //     std::cout << "True ->" << test.answers[i] << '\n';
    // }

    return 0;
}