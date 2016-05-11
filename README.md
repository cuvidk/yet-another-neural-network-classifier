#yaannc

	yaannc stands for yet another artificial neural network classifier. This provides a pretty simple API 
that anyone can use for classification. All you need to do is to collect some training set, train the network on
it and then you can provide the network an item that you want to classify and it will do it for you. This can be
achieved pretty easy, just have a look at the examples.
	To make the training process faster, a vectorized implementation of the backpropagation algorithm was
needed. I didn't write the matrix API, since there are plenty linear algebra libraries that provide very fast 
matrix operations, so [armadillo](arma.sourceforge.net) was used instead.
