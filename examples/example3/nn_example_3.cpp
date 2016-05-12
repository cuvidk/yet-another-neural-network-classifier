#include <iostream>
#include <armadillo>
#include <yaannc/neuralnetwork.h>
#include <yaannc/nnio.h>
#include <stdexcept>

int main()
{
    try
    {
	//here we create an already trained neural network from a file
        NeuralNetwork nn("xor_classifier.txt");

	//the inputs and outputs are separated in different files,
	//so we use NnIO class to load them accordingly
        arma::mat to_classify, expected_output;
        NnIO::loadSimpleData("input.txt", to_classify);
        NnIO::loadSimpleData("output.txt", expected_output);

	//we predict the outputs for all the inputs at the same time
        arma::mat actual_output = nn.predict(to_classify);

        std::cout << "For inputs:\n" << to_classify << "the neural network predicted the following:\n"
                  << actual_output << std::endl;

        std::cout << "The prediction accuracy is: " << 
		nn.getPredictionAccuracy(expected_output, actual_output) << std::endl;
    }
    catch (std::runtime_error& e)
    {
        std::cout << e.what() << std::endl;
    }
}
