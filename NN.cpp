#include "NN.h"

void normalizeData(std::vector<std::vector<double>>* data) {
    double maxDot = (*data)[0][0], minDot = (*data)[0][0];
    for (unsigned i = 0; i < (*data)[0].size(); i++) {
        for (unsigned j = 0; j < data->size(); j++) {
            maxDot = ((*data)[j][i]) > maxDot ? ((*data)[j][i]) : maxDot;
            minDot = ((*data)[j][i]) < minDot ? ((*data)[j][i]) : minDot;
        }
        for (unsigned k = 0; k < data->size(); k++) {
            if (maxDot == minDot) continue;
            (*data)[k][i] = (((*data)[k][i]) - minDot) / (maxDot - minDot);
        }
        maxDot = 0;
        minDot = (*data)[0][0];
    }
}

// Progress bar for training
void printProgress(unsigned epoch, unsigned total_epochs, double loss) {
    const int barWidth = 40;

    double progress = (double)(epoch + 1) / total_epochs;
    int percent = int(progress * 100);
    int pos = barWidth * percent / 100;

    std::string bar = "\rEpoch ";
    bar += std::to_string(epoch + 1);
    bar += "/";
    bar += std::to_string(total_epochs);
    bar += " [";

    for (int i = 0; i < barWidth; i++) {
        if (i < pos)
            bar += '=';
        else if (i == pos)
            bar += '>';
        else
            bar += ' ';
    }

    bar += "] ";
    bar += std::to_string(percent);
    bar += "% ";

    std::ostringstream loss_stream;
    loss_stream << std::fixed << std::setprecision(6) << loss;

    bar += "loss=" + loss_stream.str();

    std::cout << (percent == 99 ? GREEN : "") << bar << std::flush;
}

// Progress bar for data loading
void printProgress(size_t current, size_t total) {
    static int last_percent = -1;

    int percent = (100 * current) / total;
    if (percent == last_percent) return;
    last_percent = percent;

    const int barWidth = 40;
    int pos = barWidth * percent / 100;

    std::string bar = "\rLoading data [";

    for (int i = 0; i < barWidth; i++) {
        if (i < pos)
            bar += '=';
        else if (i == pos)
            bar += '>';
        else
            bar += ' ';
    }

    bar += "] " + std::to_string(percent + 1) + "%";
    std::cout << (percent == 99 ? GREEN : "") << bar << std::flush << RESET;
}

dataset loadData(const std::string& PATH, unsigned ANS_COUNT, unsigned OUTPUT_COUNT) {
    std::fstream dataFile(PATH, std::ios::in);
    // Container with all samples (data, answers)
    std::vector<std::vector<double>> content;
    // Buffer
    std::vector<std::vector<double>> bufDat;
    // Train part
    std::vector<std::vector<double>> data;
    std::vector<std::vector<double>> ans;
    // Test part
    std::vector<std::vector<double>> test_data;
    std::vector<std::vector<double>> test_ans;
    // Train data part %
    const double trainPercent = 0.7;

    size_t total_lines = 0;
    std::string tmp;

    while (getline(dataFile, tmp)) total_lines++;

    dataFile.clear();
    dataFile.seekg(0);

    if (dataFile.is_open()) {
        std::vector<double> row;
        std::string line;

        // Skip first line if labels
        if (!isdigit(dataFile.peek())) {
            getline(dataFile, line);
        }

        size_t line_count = 0;

        // Parsing through all lines and putting in vector<vector<double>>
        while (getline(dataFile, line)) {
            line_count++;
            printProgress(line_count, total_lines);

            std::istringstream iss(line);
            std::string token;

            while (std::getline(iss, token, ',')) {
                row.push_back(stod(token));
            }

            content.push_back(row);
            row.clear();
        }

        std::mt19937 rng(std::random_device{}());
        std::shuffle(content.begin(), content.end(), rng);

        // Separating content --> data, answers
        for (unsigned i = 0; i < content.size(); i++) {
            std::vector<double> buffer;

            for (unsigned j = 0; j < content[i].size() - ANS_COUNT; j++) {
                buffer.push_back(content[i][j]);
            }
            bufDat.push_back(buffer);

            buffer.clear();

            for (unsigned j = 0; j < OUTPUT_COUNT; j++) {
                buffer.push_back(j == content[i][content[i].size() - 1] ? 1 : 0);
            }

            // Pass 30% of dataset to testing part
            if (i < content.size() * trainPercent) {
                ans.push_back(buffer);
            } else {
                test_ans.push_back(buffer);
            }

            buffer.clear();
        }
    } else {
        std::cout << RED << "Could not open the file\n" << RESET;
        exit(1);
    }

    normalizeData(&bufDat);

    for (unsigned i = 0; i < bufDat.size(); i++) {
        if (i < bufDat.size() * trainPercent) {
            data.push_back(bufDat[i]);
        } else {
            test_data.push_back(bufDat[i]);
        }
    }

    std::cout << GREEN << "\nDataset loaded\n" << RESET;

    return dataset(data, ans, test_data, test_ans);
}

NeuralNetwork::NeuralNetwork() {};

double NeuralNetwork::actFunc(double arg, activeFunction f) {
    switch (f) {
        case activeFunction::SIGMOID:
            return 1 / (1 + expl(-arg));
            break;
        case activeFunction::RELU:
            return arg < 0 ? 0 : arg;
            break;
        case activeFunction::TANH:
            return tanh(arg);
            break;
        case activeFunction::SOFTMAX:
            //! Calculates in FeedForward()
            break;
    }
    return 0;
}

double NeuralNetwork::func_deriv(double arg, activeFunction f) {
    switch (f) {
        case activeFunction::SIGMOID:
            return arg * (1 - arg);
            break;
        case activeFunction::RELU:
            return arg < 0 ? 0 : 1;
            break;
        case activeFunction::TANH:
            return 1 - pow(arg, 2);
            break;
        case activeFunction::SOFTMAX:
            //! Calculates in FeedForward()
            break;
    }
    return 0;
}

void NeuralNetwork::setWeights() {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (unsigned i = 0; i < weights.size(); i++) {
        for (unsigned j = 0; j < weights[i].size(); j++) {
            weights[i][j] = dist(rng);
        }
    }
}

void NeuralNetwork::addLayer(unsigned neurons, activeFunction activeFunc) {
    if (neurons <= 0) {
        std::cout << RED << "Cannot add layer with <1 neurons\n" << RESET;
        exit(1);
    }

    layers.push_back({neurons, activeFunc});
}

void NeuralNetwork::print() {
    for (auto layer : layers) {
        for (int i = 0; i < layer.neurons; i++) {
            std::cout << "O  ";
        }
        std::cout << "\n-----------\n";
    }

    for (auto x : weights) {
        for (auto y : x) {
            std::cout << (y == y) << " ";
        }
        std::cout << '\n';
    }

    std::cout << "TrainRate = " << trainRate << "\nAlpha = " << alpha << '\n';
}

void NeuralNetwork::compile(double trainRate_t, double alpha_t, double epochs_t, bool bias_t, lossFunction loss_t, unsigned batch_size_t) {
    if (layers.size() < 2) {
        std::cout << RED << "Cannot compile model, less than 2 layers\n" << RESET;
        exit(1);
    }

    trainRate = trainRate_t;
    alpha = alpha_t;
    epochs = epochs_t;
    bias = bias_t;
    loss = loss_t;
    batch_size = batch_size_t;

    weights.resize(layers.size() - 1);
    for (int i = 0; i < layers.size() - 1; i++) {
        weights[i].resize(layers[i].neurons * layers[i + 1].neurons + bias * layers[i + 1].neurons);
    }

    values.resize(layers.size());
    for (unsigned i = 0; i <= weights.size(); i++) {
        values[i].resize(layers[i].neurons);
    }

    this->setWeights();
    std::cout << GREEN << "Compiling is done!\n" << RESET;
}

void NeuralNetwork::feedForward(const std::vector<double>& data) {
    // Clearing values and copy data to input layer
    for (int i = 0; i < values.size(); i++) {
        for (int j = 0; j < values[i].size(); j++) {
            values[i][j] = 0;
        }
    }
    values[0] = data;

    //! For each layer, starting with i = 1
    //! For each neuron from the i layer and after
    //! For each neuron from i-1 layer
    //! Value of current neuron = SUM of previous layer neurons * appropriate
    //! weight Then using activation function on our value

    for (unsigned i = 1; i < layers.size(); i++) {
        for (unsigned j = 0; j < layers[i].neurons; j++) {
            for (unsigned k = 0; k < layers[i - 1].neurons; k++) {
                values[i][j] += values[i - 1][k] * weights[i - 1][k * layers[i].neurons + j];
            }
            if (bias) values[i][j] += 1 * weights[i - 1][weights[i - 1].size() - layers[i].neurons + j];
            if (layers[i].activation != activeFunction::SOFTMAX) {
                values[i][j] = actFunc(values[i][j], layers[i].activation);
            }
        }
    }

    // Softmax activation (numerically stable)
    if (layers[layers.size() - 1].activation == activeFunction::SOFTMAX) {
        auto& layer = values[layers.size() - 1];
        std::vector<double> out(layer.size());

        double max_val = *std::max_element(layer.begin(), layer.end());

        double sum = 0.0;
        for (double v : layer) sum += expl(v - max_val);

        for (size_t i = 0; i < layer.size(); i++) out[i] = expl(layer[i] - max_val) / sum;

        values[layers.size() - 1] = out;
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> data = input;
    feedForward(data);
    return values[layers.size() - 1];
}

double NeuralNetwork::lossFunc(std::vector<std::vector<double>>& Ytrue, std::vector<std::vector<double>>& Ypred) {
    double losses = 0;
    switch (loss) {
        case lossFunction::MSE:
            for (unsigned i = 0; i < Ytrue.size(); i++) {
                losses += pow(Ytrue[i][0] - Ypred[i][0], 2);
            }
            losses /= Ytrue.size();
            break;
        case lossFunction::categorical_crossentropy:
            for (unsigned i = 0; i < Ytrue.size(); i++) {
                for (unsigned j = 0; j < Ytrue[i].size(); j++) {
                    double p = std::max(Ypred[i][j], 1e-15);
                    losses -= Ytrue[i][j] * log(p);
                }
            }
            losses /= Ytrue.size();
            break;
    }
    return losses;
}

void NeuralNetwork::fit(std::vector<std::vector<double>>& data, std::vector<std::vector<double>>& answers) {
    std::cout << '\n' << YELLOW;
    //*d_X | Cleans after every iteration
    std::vector<std::vector<double>> d_X;
    //* GRADs | Cleans after any iteration
    std::vector<std::vector<double>> GRADs;
    //* dW | no Cleans
    std::vector<std::vector<double>> dW;
    //* last layer index
    unsigned last = layers.size() - 1;

    d_X.resize(layers.size());
    GRADs.resize(last);
    dW.resize(weights.size());

    for (unsigned i = 0; i < d_X.size(); i++) {
        d_X[i].resize(layers[i].neurons + (i == d_X.size() - 1 ? 0 : bias));
    }

    for (int i = 0; i < last; i++) {
        GRADs[i].resize(layers[i].neurons * layers[i + 1].neurons + bias * layers[i + 1].neurons);
    }

    for (int i = 0; i < dW.size(); i++) {
        dW[i].resize(weights[i].size());
    }

    for (unsigned epoc = 0; epoc < epochs; epoc++) {
        // Vector for loss calculation
        std::vector<std::vector<double>> Ypred;

        for (unsigned set = 0; set < data.size(); set++) {
            // Feeding data to the net
            feedForward(data[set]);

            unsigned n = layers[last].neurons;

            if (layers[last].activation == activeFunction::SOFTMAX) {
                if (loss == lossFunction::categorical_crossentropy) {
                    // Softmax + CrossEntropy
                    for (unsigned i = 0; i < n; i++) d_X[last][i] = answers[set][i] - values[last][i];
                } else {
                    std::vector<double> dL_dy(n);
                    // Softmax + Another loss function
                    if (loss == lossFunction::MSE) {
                        for (unsigned i = 0; i < n; i++) dL_dy[i] = values[last][i] - answers[set][i];
                    }

                    for (unsigned i = 0; i < n; i++) {
                        double grad = 0;

                        for (unsigned j = 0; j < n; j++) {
                            double d_soft;

                            if (i == j)
                                d_soft = values[last][i] * (1 - values[last][i]);
                            else
                                d_soft = -values[last][i] * values[last][j];

                            grad += dL_dy[j] * d_soft;
                        }
                        d_X[last][i] = grad;
                    }
                }
            } else {
                for (unsigned i = 0; i < d_X[last].size(); i++) d_X[last][i] = (answers[set][i] - values[last][i]) * func_deriv(values[last][i], layers[last].activation);
            }

            // Calculating all other derives
            for (int i = last - 1; i >= 0; i--) {
                for (unsigned j = 0; j < d_X[i].size(); j++) {
                    for (unsigned k = 0; k < d_X[i + 1].size() - (i < last - 1 ? bias : 0); k++) {
                        d_X[i][j] += d_X[i + 1][k] * weights[i][j * layers[i + 1].neurons + k];
                    }
                    if (bias && (j == d_X[i].size() - 1)) {
                        d_X[i][j] *= func_deriv(1, layers[i].activation);
                    } else {
                        d_X[i][j] *= func_deriv(values[i][j], layers[i].activation);
                    }
                }
            }

            // Calculating Gradients
            for (unsigned i = 0; i < GRADs.size(); i++) {
                for (unsigned j = 0; j < GRADs[i].size(); j++) {
                    if (bias && (j >= (layers[i].neurons * layers[i + 1].neurons))) {
                        GRADs[i][j] = 1 * d_X[i + 1][j % layers[i + 1].neurons];
                    } else {
                        GRADs[i][j] = values[i][j / layers[i + 1].neurons] * d_X[i + 1][j % layers[i + 1].neurons];
                    }
                }
            }

            // Calculating dW
            for (unsigned i = 0; i < dW.size(); i++) {
                for (int j = 0; j < dW[i].size(); j++) {
                    dW[i][j] = trainRate * GRADs[i][j] + alpha * dW[i][j];
                }
            }

            // Updating weights
            for (unsigned i = 0; i < weights.size(); i++) {
                for (int j = 0; j < weights[i].size(); j++) {
                    weights[i][j] += dW[i][j];
                }
            }

            // Clearing for next iteration
            for (unsigned i = 0; i < d_X.size(); i++) {
                for (unsigned j = 0; j < d_X[i].size(); j++) {
                    d_X[i][j] = 0;
                }
            }

            for (unsigned i = 0; i < GRADs.size(); i++) {
                for (unsigned j = 0; j < GRADs[i].size(); j++) {
                    GRADs[i][j] = 0;
                }
            }
            Ypred.push_back(values[last]);
        }
        double loss_val = lossFunc(answers, Ypred);
        printProgress(epoc, epochs, loss_val);

        Ypred.clear();
    }
    std::cout << GREEN << "\nDone!\n\n" << RESET;
}