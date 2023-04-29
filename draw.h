#include <fstream>
#include <string>
#include "scalar.h"

using namespace std;

template <typename T>
void draw(scalar<T>* node) {
	if(node == nullptr) return;
	//defined general settings for the graph
    ofstream dotFile("graph.sfdp");
    dotFile << "digraph G {\n";
    dotFile << "  rankdir=LR;\n";
    dotFile << "  node[shape=record];\n";
    printNode(node, dotFile);
    dotFile << "}\n";
    dotFile.close();
    system("dot -Tpdf -o graph.pdf graph.sfdp");
    system("rm graph.sfdp");
}

template <typename T>
void printNode(scalar<T>* node, ofstream& dotFile) {
	if(node == nullptr) return;
    string label = node->_label;
    string val = to_string(node->_val);
    string grad = to_string(node->_grad);
    string nodeLabel = "{" + label + "|" + val + "|" + grad + "}";
    dotFile << node->_id << " [label=\"" << nodeLabel << "\"];\n";

	 //if and only if the node has both children
    if (node->_lchild && node->_rchild) {
        string op = node->_op;
			int op_id = node->_id*100;
			//label for the operation
        string opLabel = "{" + to_string(node->_val) + "|" + to_string(node->_grad) + "}";

			//printing the nodes
			dotFile << op_id << "[label=\"" << op << "\", shape=circle];\n";
			dotFile <<  node->_lchild->_id << " -> " << op_id << ";\n";
			dotFile << node->_rchild->_id << " -> " << op_id << ";\n";
			dotFile << op_id << " -> " << node->_id << ";\n";


			dotFile << node->_lchild->_id << " [label=\"" << to_string(node->_lchild->_id) << "\"];\n";
			dotFile << node->_rchild->_id << " [label=\"" << to_string(node->_rchild->_id) << "\"];\n";
			dotFile << node->_id << " [label=\"" << opLabel << "\"];\n";

			//recursive call on the children
			printNode(node->_lchild, dotFile);
			printNode(node->_rchild, dotFile);
    } else if (node->_lchild) {
        dotFile << node->_lchild->_id << " -> " << node->_id << ";\n";

		  //create label 
        dotFile << node->_lchild->_id << " [label=\"" << to_string(node->_lchild->_id) << "\"];\n";
        printNode(node->_lchild, dotFile);
    } else if (node->_rchild) {
        dotFile << node->_rchild->_id << " -> " << node->_id << ";\n";

        dotFile << node->_rchild->_id << " [label=\"" << to_string(node->_rchild->_id) << "\"];\n";
        printNode(node->_rchild, dotFile);
    }
}
