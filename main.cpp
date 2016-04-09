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
//    mat x = ones(2,2);
//    x.for_each( [](mat::elem_type& val) { val = log(val); } );
//    cout << x << endl;
//    mat X;
//    mat y;
//    try
//    {
//        NeuralNetworkLoader::loadUnifiedTrainingSet("input.txt", X, y);
//        cout << X << endl << y << endl;
//    }
//    catch (std::runtime_error& e)
//    {
//        std::cerr << e.what() << std::endl;
//    }
    try{
        NeuralNetwork nn({400, 25, 10});
        nn.loadLearnedWeights("weights2.txt", NNFileType::MATRIX_WEIGHTS);
        nn.loadTrainingData("input2.txt", NNFileType::UNIFIED_TRAINING_DATA);
//        arma::mat in = ones<mat>(1, 2);
//        arma::mat out = nn.predict(in);
//        cout << out << endl;
        cout << nn.computeCost() << endl;
    } catch (std::runtime_error& e){
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
