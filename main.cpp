#include <iostream>
#include <armadillo>
#include <stdexcept>

#include <vector>
#include <algorithm>
#include <iterator>

#include "neuralnetworkloader.h"
#include "neuralnetwork.h"

using namespace std;
using namespace arma;

int main()
{

    try{
        NeuralNetwork nn({2, 3, 1});
//        nn.loadLearnedWeights("weights2.txt", NNFileType::MATRIX_WEIGHTS);
        nn.loadTrainingData("input.txt", NNFileType::UNIFIED_TRAINING_DATA);
        nn.setLearningRate(0.01);
        nn.setRegularizationFactor(0);


        nn.train(1000000, 5000);

        arma::mat in = ones(1, 2);
        in(0, 1) = 0.0;
        arma::mat out = nn.predict(in);
        cout << "For input: " << in << " the output is: " << out << endl;

        arma::mat in2 = ones(1, 2);
        in2(0, 0) = 0.0;
        arma::mat out2 = nn.predict(in2);
        cout << "For input: " << in2 << " the output is: " << out2 << endl;

        arma::mat in3 = ones(1, 2);
        arma::mat out3 = nn.predict(in3);
        cout << "For input: " << in3 << " the output is: " << out3 << endl;

        arma::mat in4 = zeros(1, 2);
        arma::mat out4 = nn.predict(in4);
        cout << "For input: " << in4 << " the output is: " << out4 << endl;

        //1.6667e-08 output for 20kk epochs
        //1.6667e-09 output for 200kk epochs
    } catch (std::runtime_error& e){
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
