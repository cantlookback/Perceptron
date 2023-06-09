#include "NN.h"

NeuralNetwork::NeuralNetwork(){};

double NeuralNetwork::sigm(double arg) {
    return 1 / (1 + exp(-arg));
}

double NeuralNetwork::sigm_deriv(double arg) {
    return arg * (1 - arg);
}

void NeuralNetwork::setWeights() {
    for (unsigned i = 0; i < weights.size(); i++) {
        for (unsigned j = 0; j < weights[i].size(); j++) {
            weights[i][j] = (static_cast<double>(rand()) / RAND_MAX) * 2 - 1;
        }
    }
}

void NeuralNetwork::addLayer(unsigned neurons){
    if (neurons <= 0){
        std::cout << "Cannot add layer with <1 neurons\n";
        return;
    }
    network.first++;
    network.second.push_back(neurons);
}

void NeuralNetwork::output(){
    std::cout << values[network.first - 1] << '\n';
    return;
}

void NeuralNetwork::compile(uint64_t trainRate_t, uint64_t alpha_t, uint64_t epochs_t){
    if(network.first < 2){
        std::cout << "Cannot compile model, less than 2 layers\n";
        return;
    }
    weights.resize(network.first - 1);
    values.resize(network.first);
    for (unsigned i = 0; i <= weights.size(); i++) {
        values[i].resize(network.second[i]);
    }

    for (int i = 0; i < network.first - 1; i++) {
        weights[i].resize(network.second[i] * network.second[i + 1]);
    }

    trainRate = trainRate_t;
    alpha = alpha_t;
    epochs = epochs_t;
    
    this->setWeights();
    std::cout << "Compiling is done!\n";
}

void NeuralNetwork::print(){
    for (auto neur : network.second){
        for (int i = 0; i < neur; i++){
            std::cout << "O  ";
        }
        std::cout << "\n-----------\n";
    }
    std::cout << "TrainRate = " << trainRate << "\nAlpha = " << alpha << '\n';
}

void NeuralNetwork::feedForward(std::vector<double>* data) {
    //Copy data to input layer
    for (int i = 0; i < values.size(); i++){
        for (int j = 0; j < values[i].size(); j++){
            values[i][j] = 0;
        }
    }

    values[0] = *data;

    //!For each layer, starting with i = 1
    //!For each neuron from the i layer and beyond
    //!For each neuron from i-1 layer
    //!Value of current neuron = SUM of previous layer neurons * appropriate weight
    //!Then using activation function on our value

    for (int i = 1; i < network.first; i++) {
        for (int j = 0; j < network.second[i]; j++) {
            for (int k = 0; k < network.second[i - 1]; k++) {
                values[i][j] += values[i - 1][k] * weights[i - 1][k * network.second[i] + j];
            }
            values[i][j] = sigm(values[i][j]);
        }
    }
}

double NeuralNetwork::MSE(double Ytrue) {
    double mse = 0;
    double Ypred;
    Ypred = values[network.first - 1][0];
    mse += pow(Ytrue - Ypred, 2);
    mse /= 1;
    return mse;
}

void NeuralNetwork::fit(std::vector<std::vector<double>>* data, std::vector<double>* answers) {
    std::cout << '\n';
    //*d_X | Cleans after every iteration
    std::vector<std::vector<double>> d_X;
    //* GRADs | Cleans after any iteration
    std::vector<std::vector<double>> GRADs;
    //* dW | no Cleans
    std::vector<std::vector<double>> dW;
    
    d_X.resize(network.first);
    GRADs.resize(network.first - 1);
    dW.resize(weights.size());

    for (unsigned i = 0; i < d_X.size(); i++) {
        d_X[i].resize(network.second[i]);
    }
    for (int i = 0; i < network.first - 1; i++) {
        GRADs[i].resize(network.second[i] * network.second[i + 1]);
    }
    for (int i = 0; i < dW.size(); i++){
        dW[i].resize(weights[i].size());
    }

    trainRate = 1; //Eta
    alpha = 0.1;
    
    for (unsigned epoc = 0; epoc < epochs; epoc++) {

        for (unsigned set = 0; set < data->size(); set++) {
            //Feeding data to the net
            feedForward(&(*data)[set]);

            //Calculating the derives for output layer
            for (unsigned i = 0; i < d_X[network.first - 1].size(); i++) {
                d_X[network.first - 1][i] = ((*answers)[set] - values[network.first - 1][i]) * sigm_deriv(values[network.first - 1][i]);
            }

            //Calculating all other derives
            for (int i = network.first - 2; i >= 0; i--) {
                for (unsigned j = 0; j < d_X[i].size(); j++) {
                    for (unsigned k = 0; k < d_X[i + 1].size(); k++) {
                        d_X[i][j] += d_X[i + 1][k] * weights[i][k * (network.second[i + 1] - 1) + j];
                    }
                    d_X[i][j] *= sigm_deriv(values[i][j]);
                }
            }
            
            //Calculating Gradients
            for (unsigned i = 0; i < GRADs.size(); i++) {
                for (unsigned j = 0; j < GRADs[i].size(); j++) {
                    GRADs[i][j] = values[i][j % values[i].size()] * d_X[i + 1][int(j / values[i].size())];
                }
            }

            //Calculating dW
            for (unsigned i = 0; i < dW.size(); i++){
                for (int j = 0; j < dW[i].size(); j++){
                    dW[i][j] = trainRate * GRADs[i][j] + alpha * dW[i][j];
                }
            }
            //Updating weights
            for (unsigned i = 0; i < weights.size(); i++){
                for (int j = 0; j < weights[i].size(); j++){
                    weights[i][j] += dW[i][j];
                }
            }

            //Clearing for next iteration
            for (unsigned i = 0; i < d_X.size(); i++){
                for (unsigned j = 0; j < d_X[i].size(); j++){
                    d_X[i][j] = 0;
                }
            }

            for (unsigned i = 0; i < GRADs.size(); i++){
                for (unsigned j = 0; j < GRADs[i].size(); j++){
                    GRADs[i][j] = 0;
                }
            }
            std::cout << '\r' << epoc << " Epoch, MSE = " << MSE((*answers)[set]) << std::flush;
        }
    }
    std::cout << '\n';
}

