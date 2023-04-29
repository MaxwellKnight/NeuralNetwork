#include <iostream>
#include "draw.h"
#include "MLP.h"
using namespace std;

#define N 3

template <typename T> vector<vector<scalar<T>*>> parse_input(vector<vector<T>> input);
template <typename T> scalar<T>* MSE(vector<scalar<T>*> &guess, vector<scalar<T>*> &ypred);
template <typename T> void free(scalar<T>* root);

int main() {
	int epoch = 300; double learning_rate = 0.1;
	//data
	vector<vector<double>> data = {{1, 0, 3}, {0, -1, -2} ,{5, 9, 0}}; 
	vector<vector<scalar<double>*>> xi = parse_input(data);

	//predictions should be made by the model
	vector<scalar<double>*> ypred = {new scalar<double>(-1, "a"), new scalar<double>(1, "b"), new scalar<double>(-1, "c")}; 

	vector<int> dims = {N, 4, 4, 1}; //# inputs in each layer
	NeuralNetwork<double> model(dims); //model
	vector<scalar<double>*> guess;
	
	for(int i = 0; i < epoch; i++){
		for(auto g: guess) delete g;
		guess.clear();

		//reset gradients
		for(auto p: model.parameters()) p->_grad = 0;

		//forward pass
		for(int j = 0; j < data.size(); j++){
			vector<scalar<double>*> n = model.forward(xi[j]);
			guess.push_back(n[0]);
		}

		//calculate loss using Mean Sqaured Error
		scalar<double>* loss = MSE(guess, ypred);
		loss->backward();
		cout << loss->_val << endl;

		//updating the weights
		for(auto p: model.parameters())
			p->_val += -1 * learning_rate * p->_grad;
	}
	for(auto g: guess) cout << *g << endl;


	return 0;
}

//least mean square algorithms for calculating the loss
template<typename T>
scalar<T>* MSE(vector<scalar<T>*>& guess, vector<scalar<T>*>& ypred) {
	scalar<T>* n = new scalar<T>(guess.size(), "n", "", nullptr, nullptr);
	scalar<T>* sum = new scalar<T>(0, "s");
	scalar<T>* diff = nullptr, *sq_diff = nullptr;

	for (int i = 0; i < n->_val; i++) {
		diff = new scalar<T>(*guess[i] - *ypred[i]);
		sq_diff = new scalar<T>(power(*diff, 2));
		sum = new scalar<T>(*sum + *sq_diff);
	}
	// Calculate the mean of the squared differences
	scalar<T>* mean = new scalar<T>(*sum / *n);
	return mean;
}

//parse the input data into a scalar vector
template <typename T>
vector<vector<scalar<T>*>> parse_input(vector<vector<T>> input){
	vector<vector<scalar<T>*>> out;
	out.resize(input.size());
	for(int i = 0; i < input.size(); i++){
		for(int j = 0; j < input[i].size(); j++)
			out[i].push_back(new scalar<T>(input[i][j], "x" + to_string(i) + "," + to_string(j)));
	}

	return out;
}