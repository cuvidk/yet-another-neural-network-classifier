#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <vector>
#include <armadillo>
#include <algorithm>
#include <iterator>
#include <math.h>

#include "neuralnetworkloader.h"
#include "invalidinputexception.h"
#include "nnfiletype.h"

class NeuralNetwork
{
private:
    arma::mat m_X;
    arma::mat m_y;
    std::vector<int> m_numNeuronsOnLayer;    
    std::vector<arma::mat> m_theta;
    unsigned int m_numLayers;

public:
    NeuralNetwork(std::initializer_list<int> numNeuronsOnLayer);
    void loadLearnedWeights(const std::string& fileName, NNFileType fileType);
    void randomlyInitWeights();
};

#endif // NEURALNETWORK_H
