#include <iostream>
#include "NN.h"

using namespace std;

int main(){
    vector<vector<double>> data = {{0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}};
    vector<double> ans = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    vector<double> test;

    NeuralNetwork net;

    net.addLayer(2);
    net.addLayer(3);
    net.addLayer(3);
    net.addLayer(1);

    net.compile(1, 0.1, 1500);

    net.fit(&data, &ans);

    while (true){

        double a, b;
        std::cout << "Input a b >>";
        std::cin >> a >> b;
        test = {a, b};

        net.feedForward(&test);

        net.output();
    
    }

    return 0;
}