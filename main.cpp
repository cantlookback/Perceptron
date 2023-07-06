#include <iostream>
#include "NN.h"

using namespace std;

int main(){
    dataset samples = loadData("C:/Perceptron/data/IrisTrain3.csv", 1, 3);
    //Iris -- 0.7, 0.1, 1000, 1 || [4, 8, 4, 1]

    unsigned INPUT_SIZE = samples.data[0].size();

    NeuralNetwork net;

    net.addLayer(INPUT_SIZE);
    net.addLayer(8, SIGMOID);
    net.addLayer(4, SIGMOID);
    net.addLayer(3, SOFTMAX);

    net.compile(0.7, 0.1, 1000, 1, categorical_crossentropy);

    net.fit(&samples.data, &samples.answers);

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

    //? DATA TEST MODULE

    for (unsigned i = 0; i < samples.test_data.size(); i++){
        std::cout << "Row " << i << " testing..." << '\n';

        net.feedForward(&(samples.test_data[i]));

        std::cout << "Got -->" << *net.getOut() << '\n';
        std::cout << "True ->" << samples.test_answers[i] << '\n';
    }

    return 0;
}