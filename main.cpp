#include <iostream>

#include "NN.h"

int main() {
    json cfg = loadConfig("config.json");

    std::string filepath = cfg["dataset"]["filepath"];

    int answerSize = cfg["dataset"]["answerSize"];

    int classesCount = cfg["dataset"]["classesCount"];

    dataset samples = loadData(filepath, answerSize, classesCount);

    NeuralNetwork net;

    unsigned INPUT_SIZE = samples.data[0].size();

    net.addLayer(INPUT_SIZE);

    for (auto& layer : cfg["layers"]) {
        int neurons = layer["neurons"];
        std::string act = layer["activation"];

        net.addLayer(neurons, parseActivation(act));
    }
    lossFunction loss = parseLoss(cfg["network"]["loss"]);
    
    net.compile(cfg["network"]["trainRate"], cfg["network"]["alpha"], cfg["network"]["epochs"], cfg["network"]["bias"], loss, cfg["network"]["batch_size"]);

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

        // std::cout << "Got --> " << pred << std::endl;
        // std::cout << "True --> " << true_ans << std::endl << std::endl;
    }

    double accuracy = (double)correct / samples.test_data.size();

    std::cout << RESET << "--------------------" << std::endl << (accuracy >= 0.75 ? GREEN : RED) << "Test accuracy = " << accuracy * 100 << "%" << RESET;

    return 0;
}