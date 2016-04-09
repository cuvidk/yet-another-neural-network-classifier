#ifndef NEURALNETWORKLOADER_H
#define NEURALNETWORKLOADER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <armadillo>
#include <exception>

#include "fileopenexception.h"
#include "fileformatexception.h"

class NeuralNetworkLoader
{
private:
    NeuralNetworkLoader();
    static void insertLineInMatrix(arma::mat& matrix, unsigned int lineIndex, std::string& line);

public:
    static void loadUnifiedTrainingSet(const std::string& fileName, arma::mat& inputData,
                                       arma::mat& outputData, int numNeuronsLayer1, int numNeuronsLayerL);
    static void loadLearnedWeightsMatrix(const std::string& fileName, std::vector<arma::mat>& m_theta);
};

#endif // NEURALNETWORKLOADER_H
