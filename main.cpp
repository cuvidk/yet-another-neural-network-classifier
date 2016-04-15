#include <iostream>
#include <armadillo>
#include <stdexcept>

#include <vector>
#include <algorithm>
#include <iterator>

#include <fstream>
#include <sstream>

#include "nnio.h"
#include "neuralnetwork.h"

using namespace std;
using namespace arma;

int main()
{
    try {
        NeuralNetwork nn({2, 3, 1});
        nn.setLearningRate(1.0);
        nn.trainOn("input.txt", 3000, 50);

        mat in = ones(1, 2);
        cout << nn.predict(in) << endl;
        in = zeros(1, 2);
        cout << nn.predict(in) << endl;
        in = ones(1, 2);
        in(0, 1) = 0;
        cout << nn.predict(in) << endl;
        in = ones(1, 2);
        in(0, 0) = 0;
        cout << nn.predict(in) << endl;
    } catch (std::runtime_error& e) {
        cout << e.what() << endl;
    }

    return 0;
}
