#Y.A.A.N.N.C.

	Yet Another Artificial Neural Network Classifier.

This provides a pretty simple API that anyone can use for classification. All you need to do is to 
collect some training set, train the neural network on it and then you can provide the network an item that 
you want to classify and it will do it for you. This can be achieved pretty easy, just have a look at the examples.

To make the training process faster, a vectorized implementation of the backpropagation algorithm was
needed. I didn't write the matrix API, since there are plenty linear algebra libraries that provide very fast 
matrix operations, so [armadillo](arma.sourceforge.net) was used instead.

##Installation

First you'll have to clone this repository on your machine. I'll use the shorthand `{yaannc-root}` to refer to
the root directory of the cloned repo.
You'll also have to satisfy the armadillo library dependency. To do that please visit [armadillo's](arma.sourceforge.net) website and install the library that corresponds to your platform. For any platform, you'll find the 
installation details in the README.txt file that can be found in the armadillo archive that you've downloaded. 
For unix platforms the installation is detailed very well so the installation should go on pretty smooth. On the
other hand, windows installation for armadillo is pretty confusing so I added under `{yaannc-root}/install/windows/armadillo` everything you need so you can use yaannc library.

###Linux

This is for debian based systems, though for other systems should be similar.

* Open up a terminal;
* Make sure you have g++ installed. If you have a Debian based system you can run the following command
to install g++:

```bash
$ sudo apt-get install g++
```

* cd to `{yaannc-root}/install/linux` where a Makefile is located;
* Run the following command so the libyaannc.a file can be generated:
	
```bash
$ make
```

* Run the following command to install the yannc library and includes to your system:

```bash
$ sudo make install
```

* You're done. You can now try out one of the examples presented in the examples section. To compile your source
code you can use the following command:

```bash
$ g++ {your_source_files} -lyaannc -larmadillo -std=c++11
```
**IMPORTANT**: Be sure to specify -lyaannc library before -larmadillo, or else you will get linking errors.

If you want to uninstall the library from the system, in `{yaannc-root}/install/linux` run the following command:

```bash
$ sudo make uninstall
```

###Windows

I'll only cover how to configure a Visual Studio project so you can use yaannc library.

* Create a new empty Visual Studio C++ project;
* Right click on the newly created project and select `Properties`. Under `Configuration Properties->C/C++->General->Additional Include Directories` add the following:
  * the include directory of the armadillo library. If you didn't install armadillo, you can use the include
directory `{yaannc-root}/install/windows/armadillo/include`;
  * the include directory of the yaannc library which is located under `{yaannc-root}/install/windows/yaannc/include`.
* Right click on project, select `Properties`. Under `Configuration Properties->Linker->Additional Library Directories` add the following:
  * armadillo's library directory. The same as above, if you didn't install armadillo, you can use the directory
`{yaannc-root}/install/windows/armadillo/lib`, but take note that this directory includes the library files for 
x64 configurations only, so in the next step you will also have to use the x64 yaannc library files. If you 
want x86 lib files, you will have to get them yourself.
  * the yaannc library directory, located under `{yaannc_root}/install/windows/yaannc/lib`. You can further see
that there are available lib folders for x86 and for x64, so choose only the one that you need.
* Right click on project, select `Properties`. Under `Configuration Properties->Linker->Input->Additional Dependencies` add all the .lib files present in the directory selected at the previous step.
* Now you should be able to compile your project, so check the examples section and make a test. In order to run
it you will need to add the .dll files that come with armadillo library (e.g. blas-win64-MT.dll lapack-win64-MT.dll) in the Debug / Release folder generated by Visual Studio after building your project, or alternatively add
the folder that contains them to PATH environment variable. You're done.

##Examples

We will be focusing on how to train a XOR classifier since it's the easiest example to start with. XOR is a
logical operator that outputs true only when inputs differ. Let's consider true = 1 and false = 0. So the XOR
truth table would look like this:

```
A	B	XOR
0	0	0
0	1	1
1	0	1
1	1	0
```

This truth table is our training data. Let's throw it in a file called `training_data.txt` and after that load
it using the yaannc API. The structure of the training file should respect the following format:

```
number_of_examples number_of_features number_of_labels
feature(1)(1) feature(2)(1) ... feature(number_of_features)(1)
label(1)(1) label(2)(1) ... label(number_of_labels)(1)
...
feature(1)(number_of_examples) ... feature(number_of_features)(number_of_examples)
label(1)(number_of_examples) ... label(number_of_labels)(number_of_examples)

```

So our `training_data.txt` should look as follows:

```
4 2 1
0 0
0
0 1
1
1 0
1
1 1
0
```
**IMPORTANT**: Be sure not to end the line for the last label, or an exception will be raised.

Now let's write a program that creates a neural network consisting of 3 layers. The first layer will have 2
neurons on it, since we have 2 inputs for XOR (A and B). For the purpose of this example, the second layer (the
hidden one) will contain 3 neurons, but you really can experiment on this one. And the 3rd layer will contain 1
neuron (the output neuron) that should carry the output of our XOR operator. To do this we will write the
following line of code:

```C++
NeuralNetwork nn({2, 3, 1});
```

Now that our neural network has been created, let's train it on our training data inside `training_data.txt`, but
before that I will set the learning rate to 1.0 since it is 0.01 by default. I'm doing that because I want my 
network to learn faster, though take care tweaking that parameter.
We will also specify that we want to train for 1000 iterations and we would like to see training reports after
each 10 iterations. We can do this as follows:

```C++
nn.setLearningRate(1.0);
nn.trainOn("training_data.txt", 1000, 10);
```

So our neural network should be trained right now, so the last thing to do is to test it. To do that I'll use
armadillo's API to create an 1 x 2 matrix containing the values 1 and 0, and I'll ask the neural network to
predict the result for that input. If the network was well trained I should obtain a value very close to 1,
which would be the expected result for such an input:

```C++
arma::mat test_input(1, 2);
test_input(0, 0) = 1;
test_input(0, 1) = 0;

arma::mat test_output = nn.predict(test_input);

std::cout << "Neural network's prediction is: " << test_output << std::endl;
```
You can experiment also with test inputs like (0, 0) or (1, 1) and you should see an output very close
to 0. Now that we know that our neural network is well trained, we can save it in a file, so next
time when we want to classify some input, we will avoid losing time training our network again. We will instead
load it ready for use:

```C++
nn.exportNeuralNetwork("xor_classifier.txt");
```

To load a neural network from a file we can do the following:

```C++
NeuralNetwork nn("xor_classifier.txt");
```

or

```C++
NeuralNetwork nn({3, 2, 1});
nn.loadWeights("xor_classifier.txt");
```

To put it together:

```C++
#include <iostream>
#include <armadillo>
#include <yaannc/neuralnetwork.h>
#include <stdexcept>

int main()
{
  try
  {
    NeuralNetwork nn({2, 3, 1});
    nn.setLearningRate(1.0);
    nn.trainOn("training_data.txt", 1000, 10);

    arma::mat test_input(1, 2);
    test_input(0, 0) = 1;
    test_input(0, 1) = 0;

    arma::mat test_output = nn.predict(test_input);

    std::cout << "Neural network's prediction for input " << test_input << "is: " << test_output << std::endl;

    nn.exportNeuralNetwork("xor_classifier.txt");
  }
  catch (std::runtime_error& e)
  {
    std::cout << e.what() << std::endl;
  }
}
``` 

###More

yaannc contains a special class called NnIO which stands for Neural Network Input Output, that can handle data
loading. So you can load for example training data without directly training on it like you saw in the previous
example. You can still train on the loaded data afterwards, yaannc API providing a method to do that. Let's load
the the data inside `training_data.txt` file using NnIO.

```C++
#include <armadillo>
#include <yaannc/nnio.h>
#include <yaannc/neuralnetwork.h>

int main()
{
  try
  {
    arma::mat features, labels;
    NnIO::loadUnifiedData("training_data.txt", features, labels);

    NeuralNetwork nn({2, 3, 1});
    nn.setLearningRate(1.0);
    
    //you can use this alternative method to train on pre-loaded data
    nn.trainOn(features, labels, 1000, 1);
  }
  catch (std::runtime_error& e)
  {
    std::cout << e.what() << std::endl;
  }
}
```

NnIO also provides a method to load features / labels sitting in separate files. The file format should be:

```
number_rows number_columns
item(1)(1) ... item(1)(number_columns)
...
item(number_rows)(1) ... item(number_rows)(number_columns)
```
**IMPORTANT**: Be sure not to end the last line, or an exception will be raised.

So for file `input.txt`:
```
2 3
0 1 1
1 1 0
```
we could do the following:

```C++
arma::mat input;
NnIO::loadSimpleData("input.txt", input);
```

NeuralNetwork class also provides a method to set the regularization factor which is 0.0 by default:

```C++
nn.setRegularizationFactor(0.01);
```

There's also a usefull method that computes the prediction acurracy of your neural network. What you need to
do is to pass in the expected output and the actual output obtained by running the network on an example:

```C++
arma::mat to_classify, expected_output;
NnIO::loadUnifiedData("test_data.txt", to_classify, expected_output);
arma::mat actual_output = nn.predict(to_classify);
double accuracy = nn.getPredictionAccuracy(expected_output, actual_output);

std::cout << "The prediction accuracy is: " << accuracy << std::endl;
```
For examples, please see `{yaannc-root}/examples`.

##Author

Tiperciuc Corvin
