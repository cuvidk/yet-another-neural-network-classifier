#YAANNC

	Yet Another Artificial Neural Network Classifier.

	This provides a pretty simple API that anyone can use for classification. All you need to do is to 
collect some training set, train the neural network on it and then you can provide the network an item that 
you want to classify and it will do it for you. This can be achieved pretty easy, just have a look at the examples.
	To make the training process faster, a vectorized implementation of the backpropagation algorithm was
needed. I didn't write the matrix API, since there are plenty linear algebra libraries that provide very fast 
matrix operations, so [armadillo](arma.sourceforge.net) was used instead.

##Installation

	First you'll have to clone this repository on your machine. I'll use the shorthand {yaannc-root} to
refer to the root directory of the cloned repo.

###Linux

	* Open up a terminal;
	* Make sure you have g++ installed. If you have a Debian based system you can run the following command
	to install g++:

	```bash
	$ sudo apt-get install g++
	```

	* cd to {yaannc-root}/install/linux where a Makefile is located;
	* Run the following command so the libyaannc.a file can be generated:
	
	```bash
	$ make
	```

	* Run the following command to install the yannc library and includes to your system:

	```bash
	sudo make install
	```

	* You're done. If you want to uninstall it, you can use:

	```bash
	sudo make uninstall
	```
