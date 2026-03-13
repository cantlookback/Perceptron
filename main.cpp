#include <iostream>
#include "NN.h"

#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define RESET   "\033[0m"

using namespace std;

int main(){
    dataset samples = loadData("C:/pomoika/Perceptron/data/IrisTrain3.csv", 1, 3);
    //Iris -- 0.7, 0.1, 1000, 1 || [4, 8, 4, 3]

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

    int correct = 0;

    for (unsigned i = 0; i < samples.test_data.size(); i++){
        std::cout << "Row " << i << " testing..." << '\n';

        net.feedForward(&(samples.test_data[i]));

        std::vector<double> *pred = net.getOut();
        std::vector<double> &true_ans = samples.test_answers[i];

        
        int pred_class = 0;
        int true_class = 0;
        
        for (unsigned j = 1; j < pred->size(); j++){
            if ((*pred)[j] > (*pred)[pred_class])
            pred_class = j;
        }
        
        for (unsigned j = 0; j < true_ans.size(); j++){
            if (true_ans[j] == 1){
                true_class = j;
                break;
            }
        }

        if (pred_class == true_class){
            correct++;
            std::cout << GREEN << "Got -->" << *pred << '\n';
            std::cout << "True ->" << true_ans << '\n';            
            std::cout << "Correct\n";
        } else {
            std::cout << RED << "Got -->" << *pred << '\n';
            std::cout << "True ->" << true_ans << '\n';
            std::cout << "Wrong\n";
        }

        std::cout << RESET << '\n';
    }

    double accuracy = (double)correct / samples.test_data.size();

    std::cout << "Test accuracy = " << accuracy * 100 << "%\n";

    return 0;
}