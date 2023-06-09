#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "Neural.h"
using namespace std;

vector<vector<double>> dataset;
vector<double> answers;

//For moving dataset from file to a vector<>
void makeDatasheet() {
}

//Overload for vector<> printing
template <typename T>
ostream& operator<<(ostream &os, vector<T> &values) {
    os << '[';
    for (T val : values){
        os << val << ", ";
    }
    os << ']';
    return os;
}


int main() {
    vector<vector<double>> data = {{0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}};
    vector<double> ans = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    vector<double> test;
    
    //!! Example of working XOR
    //!! Good working with 4 layers, 2 3 3 1.
    //!! Eta = 1 | Alpha = 0.1 | Epochs = 1500
    
    cout << "A simple representation of perceptron NN\n";
    
    NeuralNetwork net;
    
    while(true){
        cout << "[1] - Create and adjust settings\n"
                "[2] - Load Dataset\n"
                "[3] - Show current model\n"
                "[4] - Train model\n"
                "[5] - Use model\n"
                "[0] - Exit\n";
    
        uint16_t choice;
        cin >> choice;

        //Choosing the option
        switch(choice){
            case 0:
                return 0;
            break;
            //Create model
            case 1:
                net.setup();
            break;
            //Dataset load
            case 2:

            break;
            //Show model
            case 3:
                net.print();
            break;
            //Train model
            case 4:
                net.train(&data, &ans);
            break;
            //Use model
            case 5:
                {
                cout << "Vector of data size >> ";
                
                uint16_t size;
                cin >> size;

                cout << "Enter data for test as a vector >> ";
                
                int a = 0;
                for (uint16_t i = 0; i < size; i++){
                    cin >> a;
                    test.push_back(a);
                }

                net.feedForward(&test);

                cout << "Output >> " << net.output() << '\n';

                test.clear();
                }
            break;

            default:
                cout << "Wrong command, try again.\n";
            break;
        }
    }
    return 0;
}