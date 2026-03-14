#include "NN_utils.h"

json loadConfig(const std::string& path) {
    std::ifstream f(path);

    if (!f.is_open()) throw std::runtime_error("Cannot open config");

    json cfg;
    f >> cfg;

    return cfg;
}

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

lossFunction parseLoss(const std::string& name) {
    if (name == "mse") return lossFunction::MSE;

    if (name == "crossentropy") return lossFunction::categorical_crossentropy;

    throw std::runtime_error("Unknown loss: " + name);
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

    std::cout << (percent == 100 ? GREEN : "") << bar << std::flush;
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
    std::cout << (percent == 100 ? GREEN : "") << bar << std::flush << RESET;
}

activeFunction parseActivation(const std::string& name) {
    if (name == "sigmoid") return activeFunction::SIGMOID;

    if (name == "relu") return activeFunction::RELU;

    if (name == "tanh") return activeFunction::TANH;

    if (name == "softmax") return activeFunction::SOFTMAX;

    throw std::runtime_error("Unknown activation: " + name);
}