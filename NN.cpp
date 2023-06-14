#include "NN.h"

dataset loadData(std::string PATH, unsigned ANS_COUNT){
    std::fstream dataFile(PATH, std::ios::in);

    std::vector<std::vector<double>> content;
    std::vector<std::vector<double>> data;
    std::vector<double> ans;

    if(dataFile.is_open()){
	    std::vector<double> row;
    	std::string line;

        //Skip first line if labels
        if (!isdigit(dataFile.peek())){
            getline(dataFile, line);
        }

        //Parsing through all lines and putting in vector<vector<double>>
    	while (getline(dataFile, line)){
            std::istringstream iss(line);
            std::string token;

            while (std::getline(iss, token, ',')){   
                row.push_back(stod(token));
            }

            content.push_back(row);
            row.clear();
        }

        //Separating content --> data, answers
        for (std::vector<double> dat : content){
            std::vector<double> buffer;

            for (unsigned i = 0; i < dat.size() - ANS_COUNT; i++){
                buffer.push_back(dat[i]);
            }
            data.push_back(buffer);
            buffer.clear();

            ans.push_back(dat[dat.size() - 1]);
        }
	} else  
		std::cout << "Could not open the file\n";

    return dataset(data, ans);
}

NeuralNetwork::NeuralNetwork(){};

double NeuralNetwork::actFunc(double arg, activeFunction f){
    switch (f){
        case SIGMOID:
            return 1 / (1 + exp(-arg));
        break;
        case RELU:

        break;
        case TANH:
            return (exp(arg) - exp(-arg)) / (exp(arg) + exp(-arg));
        break;
        case SOFTMAX:

        break;
    }
    return 0;
}

double NeuralNetwork::func_deriv(double arg, activeFunction f){
    switch (f){
        case SIGMOID:
            return arg * (1 - arg);
        break;
        case RELU:

        break;
        case TANH:
            return 1 - pow(arg, 2);
        break;
        case SOFTMAX:

        break;
    }
    return 0;
}

void NeuralNetwork::setWeights() {
    for (unsigned i = 0; i < weights.size(); i++) {
        for (unsigned j = 0; j < weights[i].size(); j++) {
            weights[i][j] = (static_cast<double>(rand()) / RAND_MAX) * 2 - 1;
        }
    }
}

void NeuralNetwork::addLayer(unsigned neurons, activeFunction activeFunc){
    if (neurons <= 0){
        std::cout << "Cannot add layer with <1 neurons\n";
        return;
    }

    network.first++;
    network.second.push_back({neurons, activeFunc});
}

void NeuralNetwork::output(){
    std::cout << values[network.first - 1] << '\n';
    return;
}

void NeuralNetwork::compile(double trainRate_t, double alpha_t, double epochs_t){
    if(network.first < 2){
        std::cout << "Cannot compile model, less than 2 layers\n";
        return;
    }

    weights.resize(network.first - 1);
    values.resize(network.first);

    for (unsigned i = 0; i <= weights.size(); i++) {
        values[i].resize(network.second[i].first);
    }

    for (int i = 0; i < network.first - 1; i++) {
        weights[i].resize(network.second[i].first * network.second[i + 1].first);
    }

    trainRate = trainRate_t;
    alpha = alpha_t;
    epochs = epochs_t;

    this->setWeights();
    std::cout << "Compiling is done!\n";
}

void NeuralNetwork::print(){
    for (auto neur : network.second){
        for (int i = 0; i < neur.first; i++){
            std::cout << "O  ";
        }
        std::cout << "\n-----------\n";
    }
    std::cout << "TrainRate = " << trainRate << "\nAlpha = " << alpha << '\n';
}

void NeuralNetwork::feedForward(std::vector<double> *data) {
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
        for (int j = 0; j < network.second[i].first; j++) {
            for (int k = 0; k < network.second[i - 1].first; k++) {
                values[i][j] += values[i - 1][k] * weights[i - 1][k * network.second[i].first + j];
            }
            values[i][j] = actFunc(values[i][j], SIGMOID);
        }
    }
}

double NeuralNetwork::MSE(std::vector<double> *Ytrue, std::vector<double> *Ypred) {
    double mse = 0;
    for (unsigned i = 0; i < Ytrue->size(); i++){
        mse += pow((*Ytrue)[i] - (*Ypred)[i], 2);
    }
    mse /= Ytrue->size();
    return mse;
}

void NeuralNetwork::fit(std::vector<std::vector<double>> *data, std::vector<double> *answers) {
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
        d_X[i].resize(network.second[i].first);
    }
    for (int i = 0; i < network.first - 1; i++) {
        GRADs[i].resize(network.second[i].first * network.second[i + 1].first);
    }
    for (int i = 0; i < dW.size(); i++){
        dW[i].resize(weights[i].size());
    }

    for (unsigned epoc = 0; epoc < epochs; epoc++) {
        //Vector for MSE
        std::vector<double> Ypred;

        for (unsigned set = 0; set < data->size(); set++) {
            //Feeding data to the net
            feedForward(&(*data)[set]);

            //Calculating the derives for output layer
            for (unsigned i = 0; i < d_X[network.first - 1].size(); i++) {
                d_X[network.first - 1][i] = ((*answers)[set] - values[network.first - 1][i]) * 
                                                    func_deriv(values[network.first - 1][i], SIGMOID);
            }

            //Calculating all other derives
            for (int i = network.first - 2; i >= 0; i--) {
                for (unsigned j = 0; j < d_X[i].size(); j++) {
                    for (unsigned k = 0; k < d_X[i + 1].size(); k++) {
                        d_X[i][j] += d_X[i + 1][k] * weights[i][k * (network.second[i + 1].first - 1) + j];
                    }
                    d_X[i][j] *= func_deriv(values[i][j], SIGMOID);
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
            Ypred.push_back(values[network.first - 1][0]);
        }
        std::cout << '\r' << epoc << " Epoch, MSE = " << MSE(answers, &Ypred) << std::flush;
        Ypred.clear();
    }
    std::cout << '\n';
}