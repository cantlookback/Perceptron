#include <iostream>

#include "NN.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout
            << "Usage: program <dataset.csv> <answer_size> <classes_count>\n";
        return 1;
    }

    std::string dataset_path = argv[1];
    int answerSize = std::stoi(argv[2]);
    int classesCount = std::stoi(argv[3]);

    dataset samples = loadData(dataset_path, answerSize, classesCount);

    unsigned INPUT_SIZE = samples.data[0].size();

    NeuralNetwork net;

    net.addLayer(INPUT_SIZE);
    // net.addLayer(8, activeFunction::SIGMOID);
    net.addLayer(2, activeFunction::SIGMOID);
    net.addLayer(classesCount, activeFunction::SOFTMAX);

    net.compile(0.7, 0.1, 50, 1, lossFunction::categorical_crossentropy, 1);

    net.fit(samples.data, samples.answers);

    std::cout << "Testing" << std::endl << "-------" << std::endl;

    int correct = 0;
    for (unsigned i = 0; i < samples.test_data.size(); i++) {
        std::vector<double> pred = net.predict(samples.test_data[i]);
        std::vector<double>& true_ans = samples.test_answers[i];

        int pred_class = 0;
        int true_class = 0;

        for (unsigned j = 1; j < pred.size(); j++) {
            if (pred[j] > pred[pred_class]) pred_class = j;
        }

        for (unsigned j = 0; j < true_ans.size(); j++) {
            if (true_ans[j] == 1) {
                true_class = j;
                break;
            }
        }

        if (pred_class == true_class) {
            correct++;
            std::cout << GREEN;
        } else {
            std::cout << RED;
        }

        //std::cout << "Got --> " << pred << std::endl;
        //std::cout << "True --> " << true_ans << std::endl << std::endl;
    }

    double accuracy = (double)correct / samples.test_data.size();

    std::cout << RESET << "--------------------" << std::endl
              << (accuracy >= 0.75 ? GREEN : RED)
              << "Test accuracy = " << accuracy * 100 << "%" << RESET;

    return 0;
}