#ifndef SCALAR_H
#define SCALAR_H
#include <string>
#include <cmath>
#include <unordered_set>
#include <stack>
using namespace std;


template <typename T> class scalar;

template<typename T> void dfs(scalar<T>* node, unordered_set<scalar<T>*>& visited, stack<scalar<T>*>& stack);

template<typename T> vector<scalar<T>*> topologicalSort(scalar<T>* outputNode);

template <typename T>
class scalar{

	public:
		T _val;
		scalar<T> *_lchild, *_rchild;
		double _grad;
		string _op;
		string _label;
		function<void()> _backward; //assignable function to calculate the local derivative
		
		//properties for drawing the graph 
		int _id;
		static int _uuid;


		scalar<T>(): _val(0), _grad(0), _op(" "), _lchild(nullptr), _rchild(nullptr){_uuid++; _id = _uuid;}
		scalar<T>(T v, string label = "", string op = "", scalar<T>* lchild = nullptr, scalar<T>* rchild = nullptr):
			_val(v), _grad(0), _backward([](){}), _op(op), 
			_lchild(lchild), _rchild(rchild), _label(label) { _uuid++; _id = _uuid; }
		~scalar<T>() {_uuid--;}

		//TanH activation function
		scalar<T>* _tanh(){
			T x = _val;
			T t = (exp(2*x) - 1)/(exp(2*x) + 1);
			scalar<T>* output = new scalar<T>(t,"tanh", "tanh", this);
			output->_backward = [this, &output, t](){
				this->_grad = (1-pow(t, 2));
			};
			return output;
		}

		//backpropagating through the network
		void backward(){
			vector<scalar<T>*> nodes = topologicalSort(this);
			cout << endl;
			this->_grad = 1;
			for(auto n: nodes)
				n->_backward();
		}

		scalar<T> expo(T x){
			scalar<T> e(exp(x), "exp", "exp", this);
			e._backward = [this, e](){
				this->_grad += e._grad * e._grad;
			};
			return e;
		}

		template <typename S>
		friend scalar<T> power(scalar<T> x, S n){
			scalar<T> output(T(pow(x._val, n)), x._label, "**", &x, nullptr);
			output._grad = n * pow(x._val, n - 1);
			return output;
		}


		// operators overloading
		friend scalar<T> operator+(scalar<T>& lhs, scalar<T>& rhs){
			scalar<T> output(lhs._val + rhs._val, " " , "+", const_cast<scalar<T>*>(&lhs), const_cast<scalar<T>*>(&rhs));

			//forwarding the gradients by the chain rule
			output._backward = [&output, &lhs, &rhs]() {
				lhs._grad += output._grad;
				rhs._grad += output._grad;
			};
			return output;
		}
		friend scalar<T> operator-(scalar<T>& lhs,  scalar<T>& rhs){
			scalar<T> output(lhs._val - rhs._val, " " , "-", const_cast<scalar<T>*>(&lhs), const_cast<scalar<T>*>(&rhs));

			//forwarding the gradients by the chain rule
			output._backward = [&output, &lhs, &rhs]() {
				lhs._grad += output._grad * 1.0;
				rhs._grad += output._grad * -1.0;
			};
			return output;
		}
		friend scalar<T> operator*(scalar<T>& lhs,scalar<T>& rhs){
			scalar<T> output(lhs._val * rhs._val, " ", "*", const_cast<scalar<T>*>(&lhs), 
			const_cast<scalar<T>*>(&rhs));

			//forwarding the gradients by the chain rule
			output._backward = [&output, &lhs, &rhs](){
				lhs._grad += rhs._val * output._grad;
				rhs._grad += lhs._val * output._grad;
			};
			return output;
		}
		friend scalar<T> operator/(scalar<T>& lhs, scalar<T>& rhs){
			scalar<T> output(lhs._val / rhs._val, " ", "/", const_cast<scalar<T>*>(&lhs), 
			const_cast<scalar<T>*>(&rhs));

			output._backward = [&output, &lhs, &rhs](){
				lhs._grad += rhs._val * output._grad;
				rhs._grad += -1 * lhs._val / pow(rhs._val, 2) * output._grad; 
			};
			return output;
		}

		//overloading print operator
		friend std::ostream& operator<<(std::ostream& os, const scalar<T>& s) {
			os << "Scalar(id= " << s._id << " , data=" << s._val << ", gradient= " << s._grad;
			s._op.empty() && (os << ", op=\'" << s._op  << "\'");
			os << ", label= " + s._label + ", children= ";
			if(s._lchild) os <<  to_string(s._lchild->_id);
			if(s._rchild) os << ", " +  to_string(s._rchild->_id);
			os << ")";
			return os;
		}

};

template <typename T>
int scalar<T>::_uuid = 0;


template<typename T>
void dfs(scalar<T>* node, unordered_set<scalar<T>*>& visited, stack<scalar<T>*>& stack) {
	visited.insert(node);

	//visiting all nodes until a leaf node
	for (scalar<T>* child : {node->_lchild, node->_rchild}) {
		if (child && visited.count(child) == 0) { 
			dfs(child, visited, stack);
		}
	}
	stack.push(node);
}

template<typename T>
vector<scalar<T>*> topologicalSort(scalar<T>* outputNode) {
	vector<scalar<T>*> sorted;
	unordered_set<scalar<T>*> visited;
	stack<scalar<T>*> stack;

	dfs(outputNode, visited, stack);

	//adding the nodes in reverse order to the list
	while (!stack.empty()) {
		sorted.push_back(stack.top());
		stack.pop();
	}
	return sorted;
}

#endif