#include "neuralnetworkloader.h"

NeuralNetworkLoader::NeuralNetworkLoader() {}

void NeuralNetworkLoader::insertLineInMatrix(arma::mat& matrix, unsigned int lineIndex, std::string& line)
{
    std::istringstream iss(line);
    std::vector<double> rowToInsert {std::istream_iterator<double>{iss}, std::istream_iterator<double>{}};
    matrix.row(lineIndex) = arma::conv_to<arma::rowvec>::from(rowToInsert);
}

void NeuralNetworkLoader::loadUnifiedTrainingSet(const std::string &fileName, arma::mat &inputData,
                                                 arma::mat &outputData, int numNeuronsLayer1, int numNeuronsLayerL)
{
    std::ifstream file(fileName);

    if (file.is_open())
    {
        unsigned int numTrainingData = 0, inputLength = 0, outputLength = 0;
        std::string line;
        std::istringstream iss;

        std::getline(file, line);
        iss.str(line);
        iss >> numTrainingData >> inputLength >> outputLength;

        if (!numTrainingData || !inputLength || !outputLength ||
                numNeuronsLayer1 != inputLength || numNeuronsLayerL != outputLength)
            throw FileFormatException(fileName, NNFileType::UNIFIED_TRAINING_DATA);

        inputData = arma::ones<arma::mat>(numTrainingData, inputLength);
        outputData = arma::ones<arma::mat>(numTrainingData, outputLength);

        unsigned int crtTrainingPair = 0;
        while (crtTrainingPair < numTrainingData)
        {
            try
            {
                std::getline(file, line);
                insertLineInMatrix(inputData, crtTrainingPair, line);

                std::getline(file, line);
                insertLineInMatrix(outputData, crtTrainingPair, line);

                ++crtTrainingPair;
            }
            catch (std::exception& e)
            {
                throw FileFormatException(fileName, NNFileType::UNIFIED_TRAINING_DATA);
            }
        }
    }
    else
        throw FileOpenException(fileName);

    file.close();
}

void NeuralNetworkLoader::loadLearnedWeightsMatrix(const std::string& fileName, std::vector<arma::mat>& m_theta)
{
    std::ifstream file(fileName);

    if (file.is_open())
    {
        std::string line;
        try
        {
            for (unsigned int layer = 0; layer < m_theta.size(); ++layer)
            {
                for (unsigned int rowIndex = 0; rowIndex < m_theta[layer].n_rows; ++rowIndex)
                {
                    std::getline(file, line);
                    insertLineInMatrix(m_theta[layer], rowIndex, line);
                }
            }
        }
        catch (std::exception& e)
        {
            throw FileFormatException(fileName, NNFileType::MATRIX_WEIGHTS);
        }
    }
    else
        throw FileOpenException(fileName);

    file.close();
}

