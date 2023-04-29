#ifndef NEURON_H
#define NEURON_H
#include <vector>
#include <random>
#include <numeric>
#include "scalar.h"
using namespace std;

template <typename T>
class neuron {
	public:
		scalar<T>* _b;
		vector<scalar<T>*> _w;
		int _nin;

		//c'tor
		neuron(int nin){
			for(int i = 0; i < nin; i++){
				random_device rd;
    			mt19937 gen(rd());
    			uniform_real_distribution<> dis(-1.0, 1.0);

    			T random_double = dis(gen);
				string label = "w" + to_string(i);
				scalar<T>* w = new scalar<T>(random_double, label); //initialize the weights with random values [-1,1]
				_w.push_back(w);
			}
			_b = new scalar<T>((float)(rand()) / (float)(RAND_MAX));
			this->_nin = nin;
		}

		//parameters
		vector<scalar<T>*> parameters(){
			vector<scalar<T>*> out;
			out.clear();
			out.insert(out.begin(), _w.begin(), _w.end());
			out.push_back(_b);
			return out;
		}

		//calculating the dot product of the xi * wi
		scalar<T>* dot_product(vector<scalar<T>*>& x) {
			if (_w.size() != x.size()) {
				cout << "_w size: " << _w.size() << ", x size: " << x.size() << endl;
				throw invalid_argument("Vectors must have the same length");
			}

			vector<scalar<T>*> tmp;
			//calculating the product of the corresponding x * wi
			for (int i = 0; i < _w.size(); i += 2) {
				scalar<T>* s1 = new scalar<T>(*(_w[i]) * *(x[i]));

				if (i + 1 < _w.size()) {
						scalar<T>* s2 = new scalar<T>(*(_w[i + 1]) * *(x[i + 1]));
						s1 = new scalar<T>(*s1 + *s2);
				}
				tmp.push_back(s1);
			}

			//iterating over the tmp vector and adding pairs until there's 
			//only one element left
			while (tmp.size() > 1) {
				vector<scalar<T>*> next;
				for (int i = 0; i < tmp.size(); i += 2) {
						scalar<T>* s1 = tmp[i];
						scalar<T>* s2 = (i + 1 < tmp.size()) ? tmp[i + 1] : new scalar<T>(0);
						scalar<T>* s3 = new scalar<T>(*s1 + *s2);
						next.push_back(s3);
				}
				tmp = move(next);
			}

			//adding the bias and returning the output
			tmp[0]->_label = "T";
			scalar<T>* out = new scalar<T>(*tmp[0] + *_b);
			out->_label = "O";
			return out->_tanh();
		}
};



#endif