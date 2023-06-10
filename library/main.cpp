#include <iostream>
#include "NN.h"

using namespace std;

int main(){
    vector<vector<double>> dataset = loadData("C:/Perceptron/library/data/xor.csv");
    //tuples by 3 {0, 0, 0}, {0, 1, 1} e.t.c

    //need to make tuples like {0, 0}, {0, 1} 
    //                          {0}    , {1}
    vector<vector<double>> data;
    vector<double> ans;
    
    unsigned ANS_COUNT = 1;

    //Separating dataset --> data and answers
    for (vector<double> dat : dataset){
        vector<double> buffer;
        
        for (unsigned i = 0; i < dat.size() - ANS_COUNT; i++){
            buffer.push_back(dat[i]);
        }
        
        data.push_back(buffer);
        buffer.clear();

        ans.push_back(dat[dat.size() - 1]);
    }

    NeuralNetwork net;

    net.addLayer(2);
    net.addLayer(3);
    net.addLayer(3);
    net.addLayer(1);

    net.compile(1, 0.1, 1500);

    net.fit(&data, &ans);

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