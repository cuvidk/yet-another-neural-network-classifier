#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <armadillo>

#include "neuralnetworkloader.h"
#include "invalidinputexception.h"
#include "nnfiletype.h"

class NeuralNetwork
{
private:
    arma::mat m_X;
    arma::mat m_y;
    std::vector<int> m_numNeuronsOnLayer;
    std::vector<arma::mat> m_activationOnLayer;
    std::vector<arma::mat> m_partialOnLayer;
    std::vector<arma::mat> m_theta;
    unsigned int m_numLayers;
    double m_regularizationFactor;
    double m_learningRate;

public:
    NeuralNetwork(std::initializer_list<int> numNeuronsOnLayer);
    NeuralNetwork(std::initializer_list<int> numNeuronsOnLayer, float regularizationFactor, float learningRate);
    arma::mat predict(arma::mat& input);
    void setRegularizationFactor(double regularizationFactor);
    void setLearningRate(double learningRate);
    void loadLearnedWeights(const std::string& fileName, NNFileType fileType);
    void loadTrainingData(const std::string& fileName, NNFileType fileType);
    void train(int numIterations, int iterations_between_report);

private:
    void randomlyInitWeights();
    arma::mat sigmoid(arma::mat input);
    arma::mat sigmoidGradient(arma::mat& input);
    arma::mat logarithm(arma::mat input);
    double computeCost();
    double computeRegTerm();
    void backprop();
    void gradientDescent(std::vector<arma::mat>& gradients);
};

#endif // NEURALNETWORK_H