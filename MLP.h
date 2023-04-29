#ifndef MLP_H
#define MLP_H
#include <vector>
#include "neuron.h"

template <typename T>
class Layer {
	public:
		vector<neuron<T>> _neurons;
		vector<scalar<T>*> _outputs;
		vector<scalar<T>*> _params;

		Layer<T>(int in, int out){
			for(int i = 0; i < in; i++)
				_neurons.push_back(neuron<T>(out));
		}

		void activate(vector<scalar<T>*> x){
			scalar<T>* act;
			_outputs.clear();
			for(auto n: _neurons){
				scalar<T>* act = n.dot_product(x);
				_outputs.push_back(act);
			}
		}

		vector<scalar<T>*> parameters(){
			_params.clear();
			for(auto n: _neurons){
				vector<scalar<T>*> a = n.parameters();
				_params.insert(_params.end(), a.begin(), a.end());
			}
			return _params;
		}
};

template <typename T>
class NeuralNetwork{
	public:
		int _in;
		vector<scalar<T>*> _outs;
		vector<Layer<T>*> _layers;
		vector<scalar<T>*> _params;

		//receving number of inputs and vector contains number of outputs in each layer
		NeuralNetwork(vector<int> dims){
			_layers.push_back(new Layer<T>(dims[0], dims[0]));
			for(int i = 0; i < dims.size() - 1; i++){
				_layers.push_back(new Layer<T>(dims[i + 1], dims[i]));
			}
		}

		//forward pass
		vector<scalar<T>*> forward(vector<scalar<T>*> x){
			_outs.swap(x);
			for(auto layer: _layers){
				layer->activate(_outs);
				_outs.swap(layer->_outputs);
			}
			return _outs;
		}

		vector<scalar<T>*> parameters(){
			_params.clear();
			for(auto layer: _layers){
				vector<scalar<T>*> a = layer->parameters();
				_params.insert(_params.end(), a.begin(), a.end());
			}
			return _params;
		}
};

#endif