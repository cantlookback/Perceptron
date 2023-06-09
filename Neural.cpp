#include "Neural.h"

NeuralNetwork::NeuralNetwork(){
    
}

void NeuralNetwork::setup(){
    //MENU 2
    while(true){
    cout << "[1] - Create new Model\n"
            "[2] - Adjuist Layers and Neurons\n"
            "[3] - Adjust TrainRate and Alpha\n"
            "[0] - Return\n";

        uint16_t choice;
        cin >> choice;

        switch(choice){
            case 0:
                return;
            break;
            //Creating model
            case 1:
                cout << "Enter number of Layers >> ";
                cin >> network.first;

                weights.resize(network.first - 1);
                values.resize(network.first);
            
                cout << "Enter the number of Neurons on each Layer (Input, Hidden, Output): \n";
                
                //Setting the number of neurons on each layer
                for (unsigned i = 0; i <= weights.size(); i++) {
                    cout << "Layer " << i + 1 << " >> ";
                    int neur;
                    cin >> neur;
                    network.second.push_back(neur);
                    values[i].resize(neur);
                }

                //Seting the number of weights on each layer
                for (int i = 0; i < network.first - 1; i++) {
                    weights[i].resize(network.second[i] * network.second[i + 1]);
                }

                cout << "Enter TrainRate >> ";
                cin >> trainRate;
                cout << "Enter Alpha >> ";
                cin >> alpha;
                cout << "Enter number of Epochs >> ";
                cin >> epochs;
                
                this->setWeights();
            break;
            //Adjusting layers&neurons
            case 2:
                cout << "Enter number of Layers >> ";
                cin >> network.first;

                cout << "Enter the number of Neurons on each Layer: \n";
                
                //Setting the number of neurons on each layer
                for (unsigned i = 0; i <= weights.size(); i++) {
                    cout << "Layer " << i + 1 << " >> ";
                    int neur;
                    cin >> neur;
                    network.second.push_back(neur);
                    values[i].resize(neur);
                }

                //Seting the number of weights on each layer
                for (int i = 0; i < network.first - 1; i++) {
                    weights[i].resize(network.second[i] * network.second[i + 1]);
                }
            
                this->setWeights();
            break;
            //Adjusting LR&A
            case 3:
                cout << "Enter TrainRate >> ";
                cin >> trainRate;
                cout << "Enter Alpha >> ";
                cin >> alpha;
                cout << "Enter number of Epochs >> ";
                cin >> epochs;
            break;
            default:
                cout << "Wrong command, try again.\n";
            break;
        }

    }
}

void NeuralNetwork::print() {
    

    for (auto neur : network.second){
        for (int i = 0; i < neur; i++){
            cout << "O  ";
        }
        cout << "\n-----------\n";
    }

    cout << "TrainRate = " << trainRate << "\nAlpha = " << alpha << '\n';

/*
    cout << "----------------------------\n   NUMBER OF LAYERS\n|";
    cout << network.first;
    cout << "|\n----------------------------\n   NUMBER OF NEURONS IN EACH LAYER\n|";
    for (auto neur : network.second) {
        cout << neur << '|';
    }
    cout << "\n----------------------------\n   NUMBER OF WEIGHT LAYERS\n|";
    cout << weights.size();
    cout << "|\n----------------------------\n   ALL WEIGHTS\n|";
    for (auto l : weights) {
        for (auto w : l) {
            cout << w << '|';
        }
        cout << "\n|";
    }
    
    cout << "----------------------------\n   VALS\n|";
    for (auto layer : values) {
        for (auto val : layer) {
            cout << val << '|';
        }
        cout << "\n|";
    }

    cout << "\n OUTPUT VALUE: " << values[network.first - 1][0] << endl;
*/
}

vector<double>& NeuralNetwork::output(){
    return values[network.first - 1];
}

double NeuralNetwork::sigm(double arg) {
    return 1 / (1 + exp(-arg));
}

double NeuralNetwork::sigm_deriv(double arg) {
    return arg * (1 - arg);
}

void NeuralNetwork::feedForward(vector<double>* data) {
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

void NeuralNetwork::setWeights() {
    for (unsigned i = 0; i < weights.size(); i++) {
        for (unsigned j = 0; j < weights[i].size(); j++) {
            weights[i][j] = (static_cast<double>(rand()) / RAND_MAX) * 2 - 1;
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

void NeuralNetwork::train(vector<vector<double>>* data, vector<double>* answers) {
    cout << '\n';
    //*d_X | Cleans after every iteration
    vector<vector<double>> d_X;
    //* GRADs | Cleans after any iteration
    vector<vector<double>> GRADs;
    //* dW | no Cleans
    vector<vector<double>> dW;
    
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
/*
            cout << "----------------------------\n   d_X   \n|";
            for (auto Gs : d_X){
                for (auto g : Gs){
                    cout << g << '|';
                }
                cout << "\n|";
            }

            cout << "----------------------------\n   GRADS   \n|";
            for (auto Gs : GRADs){
                for (auto g : Gs){
                    cout << g << '|';
                }
                cout << "\n|";
            }
            G{0,0} = v{0,0} * d{1,0}
            G{0,1} = v{0,1} * d{1,0}
            G{0,2} = v{0,0} * d{1,1}
            G{0,3} = v{0,1} * d{1,1}
            G{0,4} = v{0,0} * d{1,2}
            G{0,5} = v{0,1} * d{1,2}
            G{1,0} = v{1,0} * d{2,0}
            G{1,1} = v{1,1} * d{2,0}
            G{1,2} = v{1,2} * d{2,0}
*/
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
            cout << '\r' << epoc << " Epoch, MSE = " << MSE((*answers)[set]) << flush;
        }
    }
    cout << '\n';
}
