#include <iostream>
#include "NN.h"

using namespace std;

int main(){
    dataset train = loadData("C:/Perceptron/data/IrisTrain.csv", 1);
    //Iris -- 0.7, 0.1, 1000, 1 || [4, 8, 4, 1]

    unsigned INPUT_SIZE = train.data.size();

    NeuralNetwork net;

    net.addLayer(INPUT_SIZE, SIGMOID);
    net.addLayer(8, SIGMOID);
    net.addLayer(4, SIGMOID);
    net.addLayer(1, SIGMOID);

    net.compile(0.7, 0.1, 1000, 1, MSE);

    net.fit(&train.data, &train.answers);


    //? DATA TEST MODULE
    dataset test = loadData("C:/Perceptron/data/IrisTest.csv", 1);

    for (unsigned i = 0; i < test.data.size(); i++){
        std::cout << "Row " << i << " testing..." << '\n';

        net.feedForward(&(test.data[i]));

        std::cout << net.getOut() << '\n';
        std::cout << "True value --> [" << test.answers[i] << "]    ";

        if (round(net.getOut()) == test.answers[i]) std::cout << "YES";

        std::cout << '\n';
    }


    // vector<double> test;
    // test.resize(INPUT_SIZE);
    // while (true){
    //     std::cout << "Input >>";
    //     for (unsigned i = 0; i < INPUT_SIZE; i++){            
    //         std::cin >> test[i];
    //     }
    //     net.feedForward(&test);
        
    //     test.clear();

    //     net.output();
    // }

    return 0;
}