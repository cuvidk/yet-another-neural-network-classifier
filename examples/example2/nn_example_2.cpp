#include <armadillo>
#include <yaannc/nnio.h>
#include <yaannc/neuralnetwork.h>

int main()
{
    try
    {
        //load the data without directly training on it
        arma::mat features, labels;
        NnIO::loadUnifiedData("training_data.txt", features, labels);

        //create our neural network
        NeuralNetwork nn({2, 3, 1});
        nn.setLearningRate(1.0);

        //you can use this alternative method to train on pre-loaded data
        nn.trainOn(features, labels, 1000, 10);
    }
    catch (std::runtime_error& e)
    {
        std::cout << e.what() << std::endl;
    }
}

