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
    net.addLayer(8, SIGMOID);
    net.addLayer(4, SIGMOID);
    net.addLayer(3, SOFTMAX);

    net.compile(0.5, 0.1, 500, 1, categorical_crossentropy);

    net.fit(&train.data, &train.answers);

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


    //vector<double> dat = {12, 12, 12, 21};
    //net.feedForward(&dat);


    //cout << *net.getOut();


    //? DATA TEST MODULE
    /*
    dataset test = loadData("C:/Perceptron/test/IrisTest3.csv", 1);

    for (unsigned i = 0; i < test.data.size(); i++){
        std::cout << "Row " << i << " testing..." << '\n';

        net.feedForward(&(test.data[i]));

        std::cout << net.getOut() << '\n';
        std::cout << "True value --> [" << test.answers[i] << "]    ";

        if (round(net.getOut()) == test.answers[i]) std::cout << "YES";

        std::cout << '\n';
    }
    */

    
    return 0;
}