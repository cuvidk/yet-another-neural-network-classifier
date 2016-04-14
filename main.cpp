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

void predict(NeuralNetwork& nn, mat in)
{
    arma::mat out = nn.predict(in); //the prediction should be 0

    uword row, col;
    out.max(row, col);
    int c = (int)col;

    int answer = -1;
    if (c == 9)
        answer = 0;
    else
        answer = c + 1;
    cout << "the predicted digit for your input is: " << answer << endl;
}

int main()
{
    return 0;
}
