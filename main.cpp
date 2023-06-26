#include <iostream>
#include "NN.h"

using namespace std;

/*
    0 - Setosa
    1 - Versicolor
    2 - Verginica
*/

int main(){
    dataset train = loadData("C:/Perceptron/data/IrisTrain3.csv", 1, 3);
    //Iris -- 0.7, 0.1, 1000, 1 || [4, 8, 4, 1]

    unsigned INPUT_SIZE = train.data[0].size();

    NeuralNetwork net;

    net.addLayer(INPUT_SIZE, SIGMOID);
    net.addLayer(10, SIGMOID);
    net.addLayer(6, SIGMOID);
    net.addLayer(3, SOFTMAX);

    net.compile(0.3, 0.2, 5000, 1, categorical_crossentropy);

    net.fit(&train.data, &train.answers);


    //? DATA TEST MODULE
    dataset test = loadData("C:/Perceptron/test/IrisTest3.csv", 1, 3);

    for (unsigned i = 0; i < test.data.size(); i++){
        std::cout << "Row " << i << " testing..." << '\n';

        net.feedForward(&(test.data[i]));

        std::cout << "Got -->" << *net.getOut() << '\n';
        std::cout << "True ->" << test.answers[i] << '\n';
    }

    // vector<double> test;
    // test.resize(INPUT_SIZE);
    // while (true){
    //     std::cout << "Input >>";
    //     for (unsigned i = 0; i < INPUT_SIZE; i++){            
    //         std::cin >> test[i];
    //     }
    //     net.feedForward(&test);

    //     net.output();
    // }

    return 0;
}