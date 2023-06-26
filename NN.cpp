#include "NN.h"

dataset loadData(std::string PATH, unsigned ANS_COUNT, unsigned OUTPUT_COUNT){
    std::fstream dataFile(PATH, std::ios::in);

    std::vector<std::vector<double>> content;
    std::vector<std::vector<double>> data;
    std::vector<std::vector<double>> ans;

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
            
            for (unsigned i = 0; i < OUTPUT_COUNT; i++){
                buffer.push_back(i == dat[dat.size() - 1] ? 1 : 0);    
            }
            ans.push_back(buffer);
            buffer.clear();
        }
	} else  
		std::cout << "Could not open the file\n";

    return dataset(data, ans);
}

template <typename T>
std::ostream& operator<<(std::ostream &os, std::vector<T> &values) {
    os << '[';
    for (unsigned i = 0; i < values.size(); i++){
        os << values[i];
        if (i != values.size() - 1) os << ", ";
    }
    os << ']';
    return os;
}

NeuralNetwork::NeuralNetwork(){};

double NeuralNetwork::actFunc(double arg, activeFunction f){
    switch (f){
        case SIGMOID:
            return 1 / (1 + expl(-arg));
        break;
        case RELU:
            return arg < 0 ? 0 : arg;
        break;
        case TANH:
            return tanh(arg);
        break;
        case SOFTMAX:
            //TODO: WIP
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
            return arg < 0 ? 0 : 1;
        break;
        case TANH:
            return 1 - pow(arg, 2);
        break;
        case SOFTMAX:
            return arg * (1 - arg);
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

std::vector<double>* NeuralNetwork::getOut(){
    return &values[network.first - 1];
}

void NeuralNetwork::print(){
    for (auto layer : network.second){
        for (int i = 0; i < layer.first; i++){
            std::cout << "O  ";
        }
        std::cout << "\n-----------\n";
    }

    for (auto x : weights){
        for (auto y : x){
            std::cout << (y == y) << " ";
        }
        std::cout << '\n';
    }

    std::cout << "TrainRate = " << trainRate << "\nAlpha = " << alpha << '\n';
}

void NeuralNetwork::compile(double trainRate_t, double alpha_t, double epochs_t, bool bias_t, lossFunction loss_t){
    if(network.first < 2){
        std::cout << "Cannot compile model, less than 2 layers\n";
        return;
    }

    trainRate = trainRate_t;
    alpha = alpha_t;
    epochs = epochs_t;
    bias = bias_t;
    loss = loss_t;

    weights.resize(network.first - 1);
    for (int i = 0; i < network.first - 1; i++) {
        weights[i].resize(network.second[i].first * network.second[i + 1].first + bias * network.second[i + 1].first);
    }
    
    values.resize(network.first);
    for (unsigned i = 0; i <= weights.size(); i++) {
        values[i].resize(network.second[i].first);
    }

    this->setWeights();
    std::cout << "Compiling is done!\n";
}

void NeuralNetwork::feedForward(std::vector<double> *data) {
    //Copy data to input layer
    for (int i = 0; i < values.size(); i++){
        for (int j = 0; j < values[i].size(); j++){
            values[i][j] = 0;
        }
    }

    values[0] = *data;

    //! For each layer, starting with i = 1
    //! For each neuron from the i layer and after
    //! For each neuron from i-1 layer
    //! Value of current neuron = SUM of previous layer neurons * appropriate weight
    //! Then using activation function on our value

    for (unsigned i = 1; i < network.first; i++) {        //Layers
        for (unsigned j = 0; j < network.second[i].first; j++) {        //Neurons on i layer
            for (unsigned k = 0; k < network.second[i - 1].first; k++) {        //Neurons on i - 1 layer
                values[i][j] += values[i - 1][k] * weights[i - 1][k * network.second[i].first + j];
            }
            if (bias) values[i][j] += 1 * weights[i - 1][weights[i - 1].size() - network.second[i].first + j];
            if (network.second[i].second != SOFTMAX){
                values[i][j] = actFunc(values[i][j], network.second[i].second);
            }
        }
    }

    if (network.second[network.first - 1].second == SOFTMAX){
        std::vector<double> out;
        for (unsigned i = 0; i < network.second[network.first - 1].first; i++){
            double sum = 0;
            for (unsigned j = 0; j < network.second[network.first - 1].first; j++){
                sum += expl(values[network.first - 1][j]);
            }
            out.push_back(expl(values[network.first - 1][i]) / sum);
        }
        values[network.first - 1] = out;
    }
}

double NeuralNetwork::lossFunc(std::vector<std::vector<double>> *Ytrue, std::vector<std::vector<double>> *Ypred){
    double losses = 0;
    switch (loss){
        case MSE:
                for (unsigned i = 0; i < Ytrue->size(); i++){
                    losses += pow((*Ytrue)[i][0] - (*Ypred)[i][0], 2);
                }
            losses /= Ytrue->size();
        break;
        case categorical_crossentropy:
            for (unsigned i = 0; i < Ytrue->size(); i++){
                for (unsigned j = 0; j < (*Ytrue)[i].size(); j++){
                    losses -= (*Ytrue)[i][j] * log((*Ypred)[i][j]);
                }
            }
            losses /= Ytrue->size();
        break;
    }
    return losses;
}


void NeuralNetwork::fit(std::vector<std::vector<double>> *data, std::vector<std::vector<double>> *answers) {
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
        d_X[i].resize(network.second[i].first + (i == d_X.size() - 1 ? 0 : bias));
    }

    for (int i = 0; i < network.first - 1; i++) {
        GRADs[i].resize(network.second[i].first * network.second[i + 1].first + bias * network.second[i + 1].first);
    }

    for (int i = 0; i < dW.size(); i++){
        dW[i].resize(weights[i].size());
    }
/*
    std::cout << "---d_X sizes---\n";
    for (auto dx : d_X){
        std::cout << "[" << dx.size() << "]" << '\n';
    }
    std::cout << "---GRADs sizes---\n";
    for (auto dx : GRADs){
        std::cout << "[" << dx.size() << "]" << '\n';
    }
    std::cout << "---dW sizes---\n";
    for (auto dx : dW){
        std::cout << "[" << dx.size() << "]" << '\n';
    }
*/

    for (unsigned epoc = 0; epoc < epochs; epoc++) {
        //Vector for MSE
        std::vector<std::vector<double>> Ypred;

        for (unsigned set = 0; set < data->size(); set++) {
            //Feeding data to the net
            feedForward(&(*data)[set]);

            //Calculating the derives for output layer
            for (unsigned i = 0; i < d_X[network.first - 1].size(); i++) {
                d_X[network.first - 1][i] = ((*answers)[set][i] - values[network.first - 1][i]) * 
                        func_deriv(values[network.first - 1][i], network.second[network.first - 1].second);
            }


            //Calculating all other derives
            for (int i = network.first - 2; i >= 0; i--) {
                for (unsigned j = 0; j < d_X[i].size(); j++) {
                    for (unsigned k = 0; k < d_X[i + 1].size() - (i < network.first - 2 ? bias : 0); k++) {
                            d_X[i][j] += d_X[i + 1][k] * weights[i][k + network.second[i + 1].first * j];
                    }
                    if (bias && (j == d_X[i].size() - 1)){
                        d_X[i][j] *= func_deriv(1, network.second[i].second);
                    } else {
                        d_X[i][j] *= func_deriv(values[i][j], network.second[i].second);
                    }
                }
            }

            //Calculating Gradients
            for (unsigned i = 0; i < GRADs.size(); i++) {
                for (unsigned j = 0; j < GRADs[i].size(); j++) {
                    if (bias && (j >= (network.second[i].first * network.second[i + 1].first))){
                        GRADs[i][j] = 1 * d_X[i + 1][j % network.second[i + 1].first];
                    } else {
                        GRADs[i][j] = values[i][j / network.second[i + 1].first] * d_X[i + 1][j % network.second[i + 1].first];
                    }
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
            Ypred.push_back(values[network.first - 1]);
        }
        std::cout << '\r' << epoc << " Epoch, loss = " << lossFunc(answers, &Ypred) << std::flush;
        Ypred.clear();
    }
    std::cout << "\nDone!\n";
}