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
        NeuralNetwork nn({2, 3, 1});
        //nn.loadLearnedWeights("weights.txt", NNFileType::MATRIX_WEIGHTS);
    } catch (std::runtime_error& e){
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
