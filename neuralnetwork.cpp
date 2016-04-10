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
    return feedForward(input, m_theta);
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

void NeuralNetwork::train(int numIterations, int iterations_between_report)
{
    //IMPLEMENT CHECKS!!!!
    double prevCost = computeCost(m_theta);
    double crtCost = prevCost;
    for (int iteration = 0; iteration < numIterations; ++iteration)
    {
        if (iteration % iterations_between_report == 0 || iteration + 1 == numIterations)
            std::cout << "Iteration: " << iteration << " | Cost: " << crtCost << std::endl;
        if (crtCost > prevCost)
        {
            std::cout << "It seems like the cost is increasing. Choose a smaller learning rate." << std::endl;
            return;
        }
        backprop();
        prevCost = crtCost;
        crtCost = computeCost(m_theta);
    }
}

void NeuralNetwork::randomlyInitWeights()
{
    arma::arma_rng::set_seed_random();
    for (unsigned int layer = 0; layer < m_numLayers - 1; ++layer)
    {
        double epsilon_init = sqrt(6) / sqrt(m_numNeuronsOnLayer[layer] + m_numNeuronsOnLayer[layer + 1]);
        arma::mat crtTheta = arma::randu(m_numNeuronsOnLayer[layer + 1], m_numNeuronsOnLayer[layer] + 1);
        crtTheta = crtTheta * 2 * epsilon_init - epsilon_init;
        m_theta.push_back(crtTheta);
    }
}

arma::mat NeuralNetwork::feedForward(arma::mat& input, std::vector<arma::mat>& theta)
{
    m_activationOnLayer.clear();
    m_partialOnLayer.clear();

    arma::mat activation = input;
    for (unsigned int layer = 0; layer < theta.size(); ++layer)
    {
        activation = arma::join_horiz(arma::ones(activation.n_rows, 1), activation);
        m_activationOnLayer.push_back(activation);

        arma::mat z = activation * theta[layer].t();
        m_partialOnLayer.push_back(z);

        activation = sigmoid(z);
    }
    m_activationOnLayer.push_back(activation);
    return activation;
}

arma::mat NeuralNetwork::sigmoid(arma::mat input) const
{
    input.for_each( [](arma::mat::elem_type& val) { val = 1 / (1 + exp(-val)); } );
    return input;
}

arma::mat NeuralNetwork::sigmoidGradient(arma::mat& input) const
{
    return sigmoid(input) % (1 - sigmoid(input));
}

arma::mat NeuralNetwork::logarithm(arma::mat input) const
{
    input.for_each( [](arma::mat::elem_type& val) { val = log(val); } );
    return input;
}

double NeuralNetwork::computeCost(std::vector<arma::mat>& theta)
{
    //IMPLEMENT CHECKS!!
    arma::mat h =  feedForward(m_X, theta);
    arma::mat h_1 = 1 - h;
    unsigned int m = m_X.n_rows;
    double cost = (-1.0 / m) * arma::accu(m_y % logarithm(h) + (1 - m_y) % logarithm(h_1));
    return cost + (m_regularizationFactor / (2.0 * m)) * computeRegTerm(theta);
}

double NeuralNetwork::computeRegTerm(std::vector<arma::mat>& theta) const
{
    double regularizationTerm = 0.0;

    for (unsigned int layer = 0; layer < theta.size(); ++layer)
    {
        arma::mat toRegularize = theta[layer].cols(1, theta[layer].n_cols - 1);
        regularizationTerm += arma::accu(toRegularize % toRegularize);
    }

    return regularizationTerm;
}

void NeuralNetwork::backprop()
{
    //IMPLEMENT CHECKS, ALSO REFACTOR
    std::vector<arma::mat> gradients;

    gradients.push_back(m_activationOnLayer[m_numLayers - 1] - m_y);

    unsigned int prevErrorIndex = 0;
    for (int layer = m_numLayers - 2; layer > 0; --layer)
    {
        arma::mat error;
        error = gradients[prevErrorIndex] * m_theta[layer].cols(1, m_theta[layer].n_cols - 1);
        error = error % sigmoidGradient(m_partialOnLayer[layer - 1]);
        gradients.push_back(error);
        ++prevErrorIndex;
    }

    int errorIndex = 0;
    for (int layer = m_numLayers - 2; layer >= 0; --layer)
    {
        gradients[errorIndex] = (1.0 / m_X.n_rows) * (gradients[errorIndex].t() * m_activationOnLayer[layer]);
        ++errorIndex;
    }

    std::reverse(gradients.begin(), gradients.end());

    for (unsigned int layer = 0; layer < m_numLayers - 1; ++layer)
    {
        int lastCol = gradients[layer].n_cols - 1;
        gradients[layer].cols(1, lastCol) += (m_regularizationFactor / m_X.n_rows) * m_theta[layer].cols(1, lastCol);
    }

    //checkGradients(gradients);
    gradientDescent(gradients);
}

void NeuralNetwork::gradientDescent(std::vector<arma::mat>& gradients)
{
    for (unsigned int layer = 0; layer < m_numLayers - 1; ++layer)
        m_theta[layer] -= m_learningRate * gradients[layer];
}

void NeuralNetwork::checkGradients(std::vector<arma::mat>& gradients)
{
    double e = 0.0001;
    std::vector<arma::mat> numericalGrads;
    for (unsigned int layer = 0; layer < gradients.size(); ++layer)
    {
        std::vector<arma::mat> thetas = m_theta;
        arma::mat grads = arma::ones(m_theta[layer].n_rows, m_theta[layer].n_cols);
        for (unsigned int i = 0; i < grads.n_rows; ++i)
        {
            for (unsigned int j = 0; j < grads.n_cols; ++j)
            {
                arma::mat thetaPlus = m_theta[layer];
                arma::mat thetaMinnus = m_theta[layer];
                thetaPlus(i, j) += e;
                thetaMinnus(i, j) -= e;

                thetas[layer] = thetaPlus;
                double loss1 = computeCost(thetas);
                thetas[layer] = thetaMinnus;
                double loss2 = computeCost(thetas);

                grads(i, j) = (loss1 - loss2) / (2 * e);
            }
        }
        numericalGrads.push_back(grads);
    }
    for (unsigned int layer = 0; layer < gradients.size(); ++layer)
    {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Numerical computed gradients" << std::endl;
        std::cout << numericalGrads[layer] << std::endl;
        std::cout << "Analytical computed gradients" << std::endl;
        std::cout << gradients[layer] << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }
}

void NeuralNetwork::printWeights() const
{
    for (int i = 0; i < m_numLayers - 1; ++i)
    {
        std::cout << m_theta[i] << std::endl;
    }
}
