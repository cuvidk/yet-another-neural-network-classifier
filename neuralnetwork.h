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
    float m_regularizationFactor;
    float m_learningRate;

public:
    NeuralNetwork(std::initializer_list<int> numNeuronsOnLayer);
    NeuralNetwork(std::initializer_list<int> numNeuronsOnLayer, float regularizationFactor, float learningRate);
    void randomlyInitWeights();
    void setRegularizationTerm(float regularizationTerm);
    void setLearningRate(float learningRate);
    void loadLearnedWeights(const std::string& fileName, NNFileType fileType);
    void loadTrainingData(const std::string& fileName, NNFileType fileType);
};

#endif // NEURALNETWORK_H
