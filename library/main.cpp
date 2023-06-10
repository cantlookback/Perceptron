#include <iostream>
#include "NN.h"

using namespace std;

int main(){
    dataset dataset = loadData("C:/Perceptron/library/data/xor.csv", 1);

    NeuralNetwork net;

    net.addLayer(2);
    net.addLayer(3);
    net.addLayer(3);
    net.addLayer(1);

    net.compile(1, 0.1, 1500);

    net.fit(&dataset.data, &dataset.answers);

    while (true){
        vector<double> test;

        double a, b;
        std::cout << "Input a b >>";
        std::cin >> a >> b;
        test = {a, b};

        net.feedForward(&test);

        net.output();
    }

    return 0;
}