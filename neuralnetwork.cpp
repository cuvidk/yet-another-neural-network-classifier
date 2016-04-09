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

arma::mat NeuralNetwork::predict(arma::mat& input)
{
    arma::mat activation = input;
    for (unsigned int layer = 0; layer < m_numLayers - 1; ++layer)
    {
        activation = arma::join_horiz(arma::ones(activation.n_rows, 1), activation);
        arma::mat z = activation * m_theta[layer].t();
        activation = sigmoid(z);
    }
    return activation;
}

void NeuralNetwork::setRegularizationFactor(double regularizationFactor)
{
    m_regularizationFactor = regularizationFactor;
}

void NeuralNetwork::setLearningRate(double learningRate)
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
        NeuralNetworkLoader::loadUnifiedTrainingSet(fileName, m_X, m_y,
            m_numNeuronsOnLayer[0], m_numNeuronsOnLayer[m_numNeuronsOnLayer.size() - 1]);
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

arma::mat NeuralNetwork::sigmoid(arma::mat& input)
{
    input.for_each( [](arma::mat::elem_type& val) { val = 1 / (1 + exp(-val)); } );
    return input;
}

arma::mat NeuralNetwork::logarithm(arma::mat& input)
{
    input.for_each( [](arma::mat::elem_type& val) { val = log(val); } );
    return input;
}

double NeuralNetwork::computeCost()
{
    arma::mat h = predict(m_X);
    arma::mat h_1 = 1 - h;
    unsigned int m = m_X.n_rows;
    double cost = (-1.0 / m) * arma::accu(m_y % logarithm(h) + (1 - m_y) % logarithm(h_1));
    return cost + (m_regularizationFactor / (2 * m)) * computeRegTerm();
}

double NeuralNetwork::computeRegTerm()
{
    double regularizationTerm = 0.0;

    for (unsigned int layer = 0; layer < m_numLayers - 1; ++layer)
    {
        arma::mat toRegularize = m_theta[layer].cols(1, m_theta[layer].n_cols - 1);
        regularizationTerm += arma::accu(toRegularize % toRegularize);
    }

    return regularizationTerm;
}
