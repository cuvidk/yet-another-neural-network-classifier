#include "fileformatexception.h"

FileFormatException::FileFormatException(const std::string& fileName, NNFileType fileType)
    :std::runtime_error(fileName + "'s file format is invalid.\n"),
      m_fileType(fileType)
{}

const char* FileFormatException::what() const throw ()
{
    std::string message;
    message += std::runtime_error::what();

    switch (m_fileType)
    {
    case NNFileType::UNIFIED_TRAINING_DATA:
        message += "Make sure that your file has the following format:\n";
        message += "    #_training_examples #_input_features(n) #_output_labels(k)\n";
        message += "    input_feature_1 input_feature_2 ... input_feature_n\n";
        message += "    output_label_1 output_label_2 ... output_label_k\n";
        message += "Also make sure that #_input_features(n) = #_of_neurons_first_layer\n";
        message += "and #_output_labels(k) = #_of_neurons_last_layer.";

        break;
    case NNFileType::IDX_MNIST_LIKE_LABELS:
        break;
    case NNFileType::IDX_MNSIT_LIKE_SAMPELS:
        break;
    case NNFileType::ONE_TRAINING_EXAMPLE_ONLY:
        break;
    case NNFileType::MATRIX_WEIGHTS:
        message += "Make sure your file contains (#_of_neuron_layers - 1) matrices, representing the learned weights.\n";
        message += "Also make sure that the matrices dimensions are picked correspondingly to the number of neurons.\n";
        message += "e.g. 2 neurons first layer, 4 neurons second layer => size(first matrix in file) = 4 x 3";
        break;
    case NNFileType::ONE_LINE_WEIGHTS:
        break;
    }

    return message.c_str();
}
