#include <iostream>
#include <armadillo>
#include <yaannc/neuralnetwork.h>
#include <stdexcept>

int main()
{
    try
    {
        //create a neural network with 2 input neurons, 3 neurons on the hidden layer
        //and 1 output neuron
        NeuralNetwork nn({2, 3, 1});

        //set the learning rate of the network to 1.0
        nn.setLearningRate(1.0);

        //train the network on training_data.txt for 1000 iterations
        //get reports each 10 iterations
        nn.trainOn("training_data.txt", 1000, 10);

        //creating out input which will be (1, 0)
        arma::mat test_input(1, 2);
        test_input(0, 0) = 1;
        test_input(0, 1) = 0;

        //predict the output for input (1, 0) and store it in test_output
        arma::mat test_output = nn.predict(test_input);

        std::cout << "Neural network's prediction is: " << test_output << std::endl;

        //save the current configuration of our network in a file
        nn.exportNeuralNetwork("xor_classifier.txt");
    }
    catch (std::runtime_error& e)
    {
        std::cout << e.what() << std::endl;
    }
}
