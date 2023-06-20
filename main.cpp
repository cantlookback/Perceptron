#include <iostream>
#include "NN.h"

using namespace std;

int main(){
    dataset dataset = loadData("C:/Perceptron/data/xor.csv", 1);

    NeuralNetwork net;

    net.addLayer(2, SIGMOID);
    net.addLayer(3, SIGMOID);
    net.addLayer(3, SIGMOID);
    net.addLayer(1, SIGMOID);

    net.compile(1, 0.1, 5000, 1);

    net.fit(&dataset.data, &dataset.answers);
    //net.print();
    while (true){
        vector<double> test = {0,0};

        std::cout << "Input >>";
        for (unsigned i = 0; i < 2; i++){
        std::cin >> test[i];
        }

        net.feedForward(&test);

        net.output();
    }

    return 0;
}