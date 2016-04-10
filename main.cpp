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
        nn.setLearningRate(3.0);
        nn.train(200000000, 500000);

        arma::mat in = ones(1, 2) - 2;
        in(0, 1) = 1.0;
        arma::mat out = nn.predict(in);
        cout << "For input: " << in << " the output is: " << out << endl;

        //1.6667e-08 output for 20kk epochs
        //1.6667e-09 output for 200kk epochs
    } catch (std::runtime_error& e){
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
