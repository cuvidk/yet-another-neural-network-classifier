#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork(std::initializer_list<int> numNeuronsOnLayer)
    :NeuralNetwork(numNeuronsOnLayer, 1.0f, 1.0f)
{}

NeuralNetwork::NeuralNetwork(std::initializer_list<int> numNeuronsOnLayer, float regularizationFactor, float learningRate)
    :m_numNeuronsOnLayer(numNeuronsOnLayer),
      m_numLayers(m_numNeuronsOnLayer.size()),
      m_regularizationFactor(regularizationFactor),
      m_learningRate(learningRate)
{
    if (m_numLayers < 2)
        throw InvalidInputException("A neural network cannot have less than 2 layers.");

    for (const auto& numNeurons : numNeuronsOnLayer)
        if (numNeurons <= 0)
            throw InvalidInputException("You can't have less than 1 neuron per layer.");

    randomlyInitWeights();
}

void NeuralNetwork::randomlyInitWeights()
{
    for (unsigned int layer = 0; layer < m_numLayers - 1; ++layer)
    {
        double epsilon_init = sqrt(6) / sqrt(m_numNeuronsOnLayer[layer] + m_numNeuronsOnLayer[layer + 1]);
        arma::mat crtTheta = arma::randu(m_numNeuronsOnLayer[layer + 1], m_numNeuronsOnLayer[layer] + 1);
        crtTheta = crtTheta * 2 * epsilon_init - epsilon_init;
        m_theta.push_back(crtTheta);
    }
}

void NeuralNetwork::setRegularizationTerm(float regularizationTerm)
{
    m_regularizationFactor = regularizationTerm;
}

void NeuralNetwork::setLearningRate(float learningRate)
{
    m_learningRate = learningRate;
}

void NeuralNetwork::loadLearnedWeights(const std::string& fileName, NNFileType fileType)
{
    if (fileType == NNFileType::MATRIX_WEIGHTS)
        NeuralNetworkLoader::loadLearnedWeightsMatrix(fileName, m_theta);
}

void NeuralNetwork::loadTrainingData(const std::string& fileName, NNFileType fileType)
{
    switch (fileType)
    {
    case NNFileType::UNIFIED_TRAINING_DATA:
        NeuralNetworkLoader::loadUnifiedTrainingSet(fileName, m_X, m_y);
        break;
    case NNFileType::IDX_MNIST_LIKE_LABELS:
        break;
    case NNFileType::IDX_MNSIT_LIKE_SAMPELS:
        break;
    case NNFileType::ONE_TRAINING_EXAMPLE_ONLY:
        break;
    default:
        throw InvalidInputException("Invalid training file type selected.");
    }
}
