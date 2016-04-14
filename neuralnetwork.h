#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <armadillo>

#include "nnio.h"
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

    //DEBUGGING
    void printWeights() const;
    arma::mat getInput(int index);
private:
    void randomlyInitWeights();
    arma::mat feedForward(arma::mat& input, std::vector<arma::mat>& theta);
    arma::mat sigmoid(arma::mat input) const;
    arma::mat sigmoidGradient(arma::mat& input) const;
    arma::mat logarithm(arma::mat input) const;
    double computeCost(std::vector<arma::mat>& theta);
    double computeRegTerm(std::vector<arma::mat>& theta) const;
    void backprop();
    void gradientDescent(std::vector<arma::mat>& gradients);
    void checkGradients(std::vector<arma::mat>& gradients);
};

#endif // NEURALNETWORK_H
